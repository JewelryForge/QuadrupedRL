import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from burl.alg.ppo import PPO
from burl.alg.ac import ActorCritic, ActorTeacher, Critic
from burl.rl.state import ExtendedObservation, Action
from burl.sim import TGEnv, EnvContainer, A1, EnvContainerMultiProcess2
from burl.utils import make_cls, timestamp, TaskParam


class OnPolicyRunner:
    def __init__(self, param=TaskParam(), log_dir='log', device='cuda'):
        self.cfg = param.train_param
        make_robot = make_cls(A1, physics_param=param.sim_param)
        make_env = make_cls(TGEnv, make_robot=make_robot,
                            sim_param=param.sim_param, render_param=param.render_param)
        self.env = EnvContainerMultiProcess2(self.cfg.num_envs, make_env)

        self.device = torch.device(device)
        actor_critic = ActorCritic(ActorTeacher(), Critic(), init_noise_std=self.cfg.init_noise_std).to(self.device)
        self.alg = PPO(actor_critic, self.cfg, device=self.device)

        # init storage and model
        self.alg.init_storage(self.cfg.num_envs, self.cfg.num_steps_per_env, (ExtendedObservation.dim,),
                              (ExtendedObservation.dim,), (Action.dim,))

        # Log
        self.log_dir = os.path.join(log_dir, timestamp())
        self.writer = None
        self.total_timesteps = 0
        self.total_time = 0
        self.current_learning_iteration = 0

    def learn(self):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        privileged_obs, obs = self.env.init_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        reward_buffer, eps_len_buffer = deque(maxlen=10), deque(maxlen=10)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + self.cfg.num_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for i in range(self.cfg.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    self.env.reset(dones)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        eps_len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += self.cfg.num_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.total_timesteps += self.cfg.num_steps_per_env * self.env.num_envs
        self.total_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        it = locs['it']
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, it)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.cfg.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], it)
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], it)
        self.writer.add_scalar('Loss/learning_rate', self.alg.cfg.learning_rate, it)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), it)
        self.writer.add_scalar('Perf/total_fps', fps, it)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], it)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], it)
        reward_buffer, eps_len_buffer = locs['reward_buffer'], locs['eps_len_buffer']
        if len(reward_buffer) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(reward_buffer), it)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(eps_len_buffer), it)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(reward_buffer), self.total_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(eps_len_buffer),
                                   self.total_time)

        string = f" \033[1m Learning iteration {it}/" \
                 f"{self.current_learning_iteration + self.cfg.num_iterations} \033[0m "

        if len(reward_buffer) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{string.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(reward_buffer):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(eps_len_buffer):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{string.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.total_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.total_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.total_time / (it + 1) * (
                               self.cfg.num_iterations - it):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
