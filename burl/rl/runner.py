import os
from collections import deque

import numpy as np
import torch
import wandb

from burl.alg import Actor, Critic, PPO
from burl.rl.state import ExteroObservation, ProprioObservation, Action, ExtendedObservation
from burl.rl.task import get_task, CentralizedTask
from burl.sim import FixedTgEnv, AlienGo, EnvContainerMp2, EnvContainer, SingleEnvContainer
from burl.utils import make_cls, g_cfg, to_dev, MfTimer, log_info


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
                    actor_obs, critic_obs, rewards, dones, infos = self.env.step(actions)
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
        logs = {'Loss/value_function': locs['mean_value_loss'],
                'Loss/surrogate': locs['mean_surrogate_loss'],
                'Loss/learning_rate': self.alg.learning_rate,
                'Perform/total_fps': fps,
                'Perform/collection time': locs['collection_time'],
                'Perform/learning_time': locs['learning_time']}
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
        log_info(f"Total Frames: {it * g_cfg.num_envs * g_cfg.storage_len}")

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
        super().__init__(
            make_env=make_cls(FixedTgEnv, make_robot=AlienGo,
                              make_task=self.task_prototype.makeDistribution(get_task(task_type))),
            make_actor=make_cls(Actor, ExteroObservation.dim, ProprioObservation.dim, Action.dim,
                                g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims,
                                g_cfg.init_noise_std),
            make_critic=make_cls(Critic, ExtendedObservation.dim, 1),
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
        self.task_prototype.updateCurricula()


class Player(object):
    def __init__(self, model_path, make_env, make_actor):
        self.env = SingleEnvContainer(make_env)
        self.actor = make_actor().to(g_cfg.dev)
        log_info(f'Loading model {model_path}')
        model_info = torch.load(model_path)
        try:
            self.actor.load_state_dict(model_info['actor_state_dict'], strict=False)
        except KeyError:
            model_state_dict = model_info['model_state_dict']
            actor_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('actor.'):
                    actor_state_dict[k.removeprefix('actor.')] = v
            actor_state_dict['log_std'] = torch.zeros_like(self.actor.log_std, device=g_cfg.dev)
            self.actor.load_state_dict(actor_state_dict)

    def play(self):
        policy = self.actor.get_policy()
        with torch.inference_mode():
            actor_obs = to_dev(self.env.init_observations()[0])

            for _ in range(20000):
                actions = policy(actor_obs)
                actor_obs, _, _, dones, info = self.env.step(actions)
                actor_obs = actor_obs.to(g_cfg.dev)

                if any(dones):
                    reset_ids = torch.nonzero(dones)
                    actor_obs_reset = self.env.reset(reset_ids)[0].to(g_cfg.dev)
                    actor_obs[reset_ids,] = actor_obs_reset
                    print('episode reward', float(info['episode_reward']))


class PolicyPlayer(Player):
    def __init__(self, model_path, task_type='basic'):
        task_prototype = CentralizedTask()

        super().__init__(
            model_path,
            make_env=make_cls(FixedTgEnv, make_robot=AlienGo,
                              make_task=task_prototype.makeDistribution(get_task(task_type))),
            make_actor=make_cls(Actor, ExteroObservation.dim, ProprioObservation.dim, Action.dim,
                                g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims)
        )
