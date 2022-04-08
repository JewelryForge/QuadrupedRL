import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import wandb

from burl.alg.ac import Actor, Critic
from burl.alg.dagger import SlidingWindow
from burl.alg.ppo import PPO
from burl.alg.student import Student
from burl.rl.task import get_task, CentralizedTask
from burl.sim.env import FixedTgEnv, robot_auto_maker
from burl.sim.motor import ActuatorNetManager
from burl.sim.parallel import EnvContainerMp2, EnvContainer, SingleEnvContainer
from burl.sim.state import ExteroObservation, RealWorldObservation, Action, ExtendedObservation, ProprioInfo
from burl.utils import make_part, g_cfg, to_dev, MfTimer, log_info, log_warn


class Accountant(object):
    def __init__(self):
        self._account = {}
        self._times = {}

    def register(self, items: dict):
        for k, v in items.items():
            v = np.asarray(v)
            self._account[k] = self._account.get(k, 0) + np.sum(v)
            self._times[k] = self._times.get(k, 0) + np.size(v)

    def query(self, key):
        return self._account[key] / self._times[key]

    def report(self):
        report = self._account.copy()
        for k in report:
            report[k] /= self._times[k]
        return report

    def clear(self):
        self._account.clear()
        self._times.clear()


class OnPolicyRunner(object):
    def __init__(self, make_env, make_actor, make_critic):
        self.env = (EnvContainerMp2 if g_cfg.use_mp else EnvContainer)(make_env, g_cfg.num_envs)
        self.alg = PPO(make_actor(), make_critic())

        self.current_iter = 0

    def learn(self):
        actor_obs, critic_obs = to_dev(*self.env.init_observations())

        reward_buffer, eps_len_buffer = deque(maxlen=g_cfg.num_envs), deque(maxlen=g_cfg.num_envs)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=g_cfg.dev)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=g_cfg.dev)
        total_iter = g_cfg.num_iterations
        accountant = Accountant()
        timer = MfTimer()
        for it in range(self.current_iter + 1, total_iter + 1):
            self.current_iter += 1
            with torch.inference_mode():
                timer.start()
                for _ in range(g_cfg.storage_len):
                    actions = self.alg.act(actor_obs, critic_obs)
                    (actor_obs, critic_obs), rewards, dones, infos = self.env.step(actions)
                    actor_obs, critic_obs, rewards, dones = to_dev(actor_obs, critic_obs, rewards, dones)
                    self.alg.process_env_step(rewards, dones, infos['time_out'].to(g_cfg.dev))
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    accountant.register(infos['reward_details'])

                    if any(dones):
                        reset_ids = torch.squeeze(dones.nonzero(), dim=1)
                        actor_obs_reset, critic_obs_reset = to_dev(*self.env.reset(reset_ids))
                        actor_obs[reset_ids,], critic_obs[reset_ids,] = actor_obs_reset, critic_obs_reset
                        reward_buffer.extend(cur_reward_sum[reset_ids].cpu().numpy().tolist())
                        eps_len_buffer.extend(cur_episode_length[reset_ids].cpu().numpy().tolist())
                        cur_reward_sum[reset_ids] = 0
                        cur_episode_length[reset_ids] = 0
                        self.on_env_reset(reset_ids)

                task_infos = infos.get('task_info', {})
                collection_time = timer.end()

                timer.start()
                # Learning step
                self.alg.compute_returns(critic_obs)
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            learning_time = timer.end()
            self.log(it, locals())
            if it % g_cfg.save_interval == 0:
                self.save(os.path.join(g_cfg.log_dir, f'model_{it}.pt'))

        self.save(os.path.join(g_cfg.log_dir, f'model_{self.current_iter}.pt'))

    def on_env_reset(self, reset_ids):
        pass

    def log(self, it, locs, width=25):
        log_info(f"{'#' * width}")
        log_info(f"Iteration {it}/{locs['total_iter']}")
        log_info(f"Collection Time: {locs['collection_time']:.3f}")
        log_info(f"Learning Time: {locs['learning_time']:.3f}")

        fps = int(g_cfg.storage_len * self.env.num_envs / (locs['collection_time'] + locs['learning_time']))
        num_frames = it * g_cfg.num_envs * g_cfg.storage_len
        logs = {'Loss/value_function': locs['mean_value_loss'],
                'Loss/surrogate': locs['mean_surrogate_loss'],
                'Loss/learning_rate': self.alg.learning_rate,
                'Perform/total_fps': fps,
                'Perform/collection time': locs['collection_time'],
                'Perform/learning_time': locs['learning_time'],
                'Perform/num_frames': num_frames}
        logs.update(self.get_policy_info())
        logs.update({f'Reward/{k}': v for k, v in locs['accountant'].report().items()})
        logs.update({f'Task/{k}': v.numpy().mean() for k, v in locs['task_infos'].items()})
        locs['accountant'].clear()
        reward_buffer, eps_len_buffer = locs['reward_buffer'], locs['eps_len_buffer']
        if len(reward_buffer) > 0:
            reward_mean, eps_len_mean = np.mean(reward_buffer), np.mean(eps_len_buffer)
            logs.update({'Train/mean_reward': reward_mean,
                         'Train/mean_episode_length': eps_len_mean}),
            log_info(f"{'Mean Reward:'} {reward_mean:.3f}")
            log_info(f"{'Mean EpsLen:'} {eps_len_mean:.1f}")
        log_info(f"Total Frames: {num_frames}")

        wandb.log(logs, step=it)

    def get_policy_info(self):
        return {}

    def save(self, path, infos=None):
        if not os.path.exists(d := os.path.dirname(path)):
            os.makedirs(d)
        torch.save({
            'actor_state_dict': self.alg.actor.state_dict(),
            'critic_state_dict': self.alg.critic.state_dict(),
            'optimizer_state_dict': self.alg.optim.state_dict(),
            'iter': self.current_iter,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.alg.critic.load_state_dict(loaded_dict['critic_state_dict'])
        if load_optimizer:
            self.alg.optim.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_iter = loaded_dict['iter']
        return loaded_dict['infos']


class PolicyTrainer(OnPolicyRunner):
    def __init__(self, task_type='basic'):
        self.task_prototype = CentralizedTask()
        if g_cfg.actuator_net:
            self.acnet_manager = ActuatorNetManager(g_cfg.actuator_net)
        else:
            self.acnet_manager = g_cfg.actuator_net
        make_robot = robot_auto_maker(actuator_net=self.acnet_manager)
        super().__init__(
            make_env=make_part(FixedTgEnv, make_robot=make_robot,
                               make_task=self.task_prototype.spawner(get_task(task_type))),
            make_actor=make_part(Actor, ExteroObservation.dim, RealWorldObservation.dim, Action.dim,
                                 g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims,
                                 g_cfg.init_noise_std),
            make_critic=make_part(Critic, ExtendedObservation.dim, 1),
        )

    def get_policy_info(self):
        std = self.alg.actor.std.cpu()
        return {  # 'Policy/freq_noise_std': std[:4].mean().item(),
            'Policy/X_noise_std': std[(0, 3, 6, 9),].mean().item(),
            'Policy/Y_noise_std': std[(1, 4, 7, 10),].mean().item(),
            'Policy/Z_noise_std': std[(2, 5, 8, 11),].mean().item()}
        # return {'Policy/freq_noise_std': std[:4].mean().item(),
        #         'Policy/X_noise_std': std[(4, 7, 10, 13),].mean().item(),
        #         'Policy/Y_noise_std': std[(5, 8, 11, 14),].mean().item(),
        #         'Policy/Z_noise_std': std[(6, 9, 12, 15),].mean().item()}

    def on_env_reset(self, reset_ids):
        self.task_prototype.update_curricula()


class TeacherPlayer(object):
    def __init__(self, model_path, task_type='basic'):
        task_proto = CentralizedTask()
        make_robot = robot_auto_maker(actuator_net=g_cfg.actuator_net)
        make_env = make_part(FixedTgEnv, make_robot, task_proto.spawner(get_task(task_type)), 'noisy_extended')
        self.env = SingleEnvContainer(make_env)
        self.policy = Actor(ExteroObservation.dim, RealWorldObservation.dim, Action.dim,
                            g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims).to(g_cfg.dev)
        model_info = torch.load(model_path)
        log_info(f'Loading model {model_path}')
        self.policy.load_state_dict(model_info['actor_state_dict'], strict=False)

    def play(self, allow_reset=True):
        policy = self.policy.get_policy()
        with torch.inference_mode():
            obs = to_dev(self.env.init_observations()[0])

            for _ in range(20000):
                actions = policy(obs)
                self.loop_callback()
                (obs,), _, dones, info = self.env.step(actions)
                obs = obs.to(g_cfg.dev)

                if any(dones) and allow_reset:
                    reset_ids = torch.nonzero(dones)
                    obs_reset = self.env.reset(reset_ids)[0].to(g_cfg.dev)
                    obs[reset_ids,] = obs_reset
                    print('episode reward', float(info['episode_reward']))

    def loop_callback(self):
        pass


class StudentPlayer(object):
    def __init__(self, model_path, task_type='basic'):
        task_proto = CentralizedTask()
        make_robot = robot_auto_maker(actuator_net=g_cfg.actuator_net)
        make_env = make_part(FixedTgEnv, make_robot, task_proto.spawner(get_task(task_type)),
                             obs_types=('noisy_proprio_info', 'noisy_realworld'))
        self.env = SingleEnvContainer(make_env)
        teacher = Actor(ExteroObservation.dim, RealWorldObservation.dim, Action.dim,
                        g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims)
        self.policy = Student(teacher).to(g_cfg.dev)
        model_info = torch.load(model_path)
        log_info(f'Loading model {model_path}')
        self.policy.load_state_dict(model_info['student_state_dict'])
        self.history = SlidingWindow(ProprioInfo.dim, 2000, g_cfg.history_len, 'cuda')

    def play(self, allow_reset=True):
        policy = self.policy.get_policy()
        with torch.inference_mode():
            proprio_info, realworld_obs = to_dev(*self.env.init_observations())

            for _ in range(20000):
                self.history.add_transition(proprio_info)
                actions = policy(self.history.get_window(), realworld_obs)
                self.loop_callback()
                (proprio_info, realworld_obs), _, dones, info = self.env.step(actions)
                proprio_info, realworld_obs = to_dev(proprio_info, realworld_obs)

                if any(dones) and allow_reset:
                    reset_ids = torch.nonzero(dones)
                    proprio_info, realworld_obs = to_dev(*self.env.reset(reset_ids))
                    print('episode reward', float(info['episode_reward']))
                    self.history.clear()

    def loop_callback(self):
        pass


def JoystickPlayer(base_player, gamepad_type='PS4'):
    class _JoystickPlayer(base_player):
        def __init__(self, model_path, task_type='basic'):
            super().__init__(model_path, task_type)
            from thirdparty.gamepad import gamepad, controllers
            if not gamepad.available():
                log_warn('Please connect your gamepad...')
                while not gamepad.available():
                    time.sleep(1.0)
            try:
                self.gamepad: gamepad.Gamepad = getattr(controllers, gamepad_type)()
            except AttributeError:
                raise RuntimeError(f'`{gamepad_type}` is not supported,'
                                   f'all {controllers.all_controllers}')
            self.gamepad.startBackgroundUpdates()
            log_info('Gamepad connected')

        @staticmethod
        def is_available():
            from thirdparty.gamepad import gamepad
            return gamepad.available()

        def loop_callback(self):
            if self.gamepad.isConnected():
                x_speed = -self.gamepad.axis('LEFT-Y')
                y_speed = -self.gamepad.axis('LEFT-X')
                steering = -self.gamepad.axis('RIGHT-X')
                steering = 1. if steering > 0.2 else -1. if steering < -0.2 else 0.
                speed_norm = math.hypot(x_speed, y_speed)
                if speed_norm:
                    self.env.unwrapped.task.cmd = (x_speed / speed_norm, y_speed / speed_norm, steering)
                else:
                    self.env.unwrapped.task.cmd = (0., 0., steering)
            else:
                sys.exit(1)

        def __del__(self):
            self.gamepad.disconnect()

    return _JoystickPlayer


JoystickTeacherPlayer = JoystickPlayer(TeacherPlayer)
JoystickStudentPlayer = JoystickPlayer(StudentPlayer)
