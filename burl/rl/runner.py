import os
import time
from collections import deque
from multiprocessing import Process, Queue, Pipe

import numpy as np
import torch
import wandb

from burl.alg.ac import ActorCritic, Actor, Critic
from burl.alg.ppo import PPO
from burl.rl.task import BasicTask, RandomCmdTask
from burl.rl.state import ExteroObservation, ProprioObservation, Action, ExtendedObservation
from burl.sim import TGEnv, A1, AlienGo, EnvContainerMultiProcess2, EnvContainer
from burl.utils import make_cls, g_cfg, to_dev, WithTimer, log_info


class Accountant:
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


class OnPolicyRunner:
    def __init__(self):
        make_robot = make_cls(A1)
        # make_robot = make_cls(AlienGo)
        make_task = make_cls(BasicTask)
        # make_task = make_cls(RandomCmdTask)
        make_env = make_cls(TGEnv, make_task=make_task, make_robot=make_robot)
        if g_cfg.use_mp:
            self.env = EnvContainerMultiProcess2(make_env, g_cfg.num_envs)
        else:
            self.env = EnvContainer(make_env, g_cfg.num_envs)
        if g_cfg.validation:
            self.eval_env = EnvContainer(make_env, 1)
        actor_critic = ActorCritic(
            Actor(ExteroObservation.dim, ProprioObservation.dim, Action.dim,
                  g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims),
            Critic(ExtendedObservation.dim, 1), g_cfg.init_noise_std).to(g_cfg.dev)
        self.alg = PPO(actor_critic)

        self.current_iter = 0

    def learn(self):
        p_obs, obs = to_dev(*self.env.init_observations())
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        reward_buffer, eps_len_buffer = deque(maxlen=10), deque(maxlen=10)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=g_cfg.dev)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=g_cfg.dev)
        total_iter = self.current_iter + g_cfg.num_iterations
        accountant = Accountant()
        timer = WithTimer()
        for it in range(self.current_iter + 1, total_iter + 1):
            with torch.inference_mode():
                timer.start()
                for _ in range(g_cfg.storage_len):
                    actions = self.alg.act(obs, p_obs)
                    p_obs, obs, rewards, dones, infos = self.env.step(actions)
                    p_obs, obs, rewards, dones = to_dev(p_obs, obs, rewards, dones)
                    self.alg.process_env_step(rewards, dones, infos['time_out'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    accountant.register(infos['reward_details'])

                    if any(dones):
                        reset_ids = torch.squeeze(dones.nonzero(), dim=1)
                        p_obs_reset, obs_reset = to_dev(*self.env.reset(reset_ids))
                        p_obs[reset_ids,], obs[reset_ids,] = p_obs_reset, obs_reset
                        reward_buffer.extend(cur_reward_sum[reset_ids].cpu().numpy().tolist())
                        eps_len_buffer.extend(cur_episode_length[reset_ids].cpu().numpy().tolist())
                        cur_reward_sum[reset_ids] = 0
                        cur_episode_length[reset_ids] = 0

                if 'difficulty' in infos:
                    difficulty = np.mean(infos['difficulty'])
                collection_time = timer.end()

                timer.start()
                # Learning step
                self.alg.compute_returns(p_obs)
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            learning_time = timer.end()
            self.log(it, locals())
            if it % g_cfg.save_interval == 0:
                self.save(os.path.join(g_cfg.log_dir, f'model_{it}.pt'))

        self.current_iter += g_cfg.num_iterations
        self.save(os.path.join(g_cfg.log_dir, f'model_{self.current_iter}.pt'))

    def log(self, it, locs, width=25):
        log_info(f"{'#' * width}")
        log_info(f"Iteration {it}/{locs['total_iter']}")
        log_info(f"Collection Time: {locs['collection_time']:.3f}")
        log_info(f"Learning Time: {locs['learning_time']:.3f}")

        fps = int(g_cfg.storage_len * self.env.num_envs / (locs['collection_time'] + locs['learning_time']))
        logs = {'Loss/value_function': locs['mean_value_loss'],
                'Loss/surrogate': locs['mean_surrogate_loss'],
                'Loss/learning_rate': self.alg.learning_rate,
                'Policy/freq_noise_std': self.alg.actor_critic.std.cpu()[:4].mean().item(),
                'Policy/X_noise_std': self.alg.actor_critic.std.cpu()[(4, 7, 10, 13),].mean().item(),
                'Policy/Y_noise_std': self.alg.actor_critic.std.cpu()[(5, 8, 11, 14),].mean().item(),
                'Policy/Z_noise_std': self.alg.actor_critic.std.cpu()[(6, 9, 12, 15),].mean().item(),
                'Perform/total_fps': fps,
                'Perform/collection time': locs['collection_time'],
                'Perform/learning_time': locs['learning_time']}
        logs.update({f'Reward/{k}': v for k, v in locs['accountant'].report().items()})
        locs['accountant'].clear()
        reward_buffer, eps_len_buffer = locs['reward_buffer'], locs['eps_len_buffer']
        if 'difficulty' in locs:
            logs.update({'Train/difficulty': locs['difficulty']}),
            log_info(f"{'Difficulty:'} {locs['difficulty']:.3f}")
        if len(reward_buffer) > 0:
            reward_mean, eps_len_mean = np.mean(reward_buffer), np.mean(eps_len_buffer)
            logs.update({'Train/mean_reward': reward_mean,
                         'Train/mean_episode_length': eps_len_mean}),
            log_info(f"{'Mean Reward:'} {reward_mean:.3f}")
            log_info(f"{'Mean EpsLen:'} {eps_len_mean:.1f}")
        log_info(f"Total Frames: {it * g_cfg.num_envs * g_cfg.storage_len}")

        wandb.log(logs, step=it)

    # def log_eval(self, it, locs, width=25):
    #     log_info(f"{'#' * width}")
    #     log_info(f"Evaluation {it}")
    #     logs = {f'Eval/{k}': v for k, v in locs['accountant'].report().items()}
    #     wandb.log(logs, step=it)

    def save(self, path, infos=None):
        if not os.path.exists(d := os.path.dirname(path)):
            os.makedirs(d)
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_iter,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_iter = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
