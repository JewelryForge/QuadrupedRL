import os
from collections import deque

import numpy as np
import torch
import wandb
from torch.optim import AdamW

from burl.alg import Actor
from burl.alg.dagger import Dagger
from burl.alg.student import Student
from burl.rl.runner import Accountant
from burl.sim.parallel import EnvContainerMp2, EnvContainer
from burl.utils import g_cfg, to_dev, MfTimer, log_info


class ImitationLearning(object):
    def __init__(self, make_env, teacher, make_student, make_alg, device):
        self.num_envs = g_cfg.num_envs
        self.device = torch.device(device)
        self.env = (EnvContainerMp2 if g_cfg.use_mp else EnvContainer)(make_env, g_cfg.num_envs)
        self.teacher: Actor = teacher.to(g_cfg.dev)
        self.student: Student = make_student(self.teacher).to(g_cfg.dev)
        self.alg: Dagger = make_alg()
        self.optim = AdamW(self.student.parameters(), lr=g_cfg.learning_rate, weight_decay=1e-2)
        self.current_epoch = 0
        self.criterion = torch.nn.MSELoss()

    def learn(self, num_epochs, num_steps_each_epoch, batch_size, repeat_learning_times,
              save_interval, log_dir, record_infos: bool):
        proprio_obs, real_world_obs, extended_obs = to_dev(*self.env.init_observations())
        self.alg.add_transitions(proprio_obs, real_world_obs, *self.teacher.get_encoded(extended_obs))
        if record_infos:
            accountant = Accountant()
            cur_reward_sum = torch.zeros(self.num_envs, device=self.device)
            cur_episode_length = torch.zeros(self.env.num_envs, device=g_cfg.dev)
            reward_buffer, eps_len_buffer = deque(maxlen=self.num_envs), deque(maxlen=self.num_envs)
        timer = MfTimer()
        for it in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch += 1
            with torch.inference_mode():
                timer.start()
                policy = self.alg.get_policy(self.teacher, self.student)
                for _ in range(num_steps_each_epoch):
                    actions = policy(extended_obs, (self.alg.get_proprio_history(), real_world_obs))
                    (proprio_obs, real_world_obs, extended_obs), rewards, dones, infos = self.env.step(actions)
                    proprio_obs, real_world_obs, extended_obs = to_dev(proprio_obs, real_world_obs, extended_obs)

                    if record_infos:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        accountant.register(infos['reward_details'])

                    if any(dones):
                        reset_ids = torch.squeeze(dones.nonzero(), dim=1)
                        proprio_obs_r, real_world_obs_r, extended_obs_r = to_dev(*self.env.reset(reset_ids))
                        proprio_obs[(reset_ids,)] = proprio_obs_r
                        real_world_obs[(reset_ids,)] = real_world_obs_r
                        extended_obs[(reset_ids,)] = extended_obs_r
                        if record_infos:
                            reward_buffer.extend(cur_reward_sum[reset_ids].cpu().numpy().tolist())
                            eps_len_buffer.extend(cur_episode_length[reset_ids].cpu().numpy().tolist())
                            cur_reward_sum[reset_ids] = 0
                            cur_episode_length[reset_ids] = 0

                    task_infos = infos.get('task_info', {})
                    encoded = self.teacher.get_encoded(extended_obs)
                    self.alg.add_transitions(proprio_obs, real_world_obs, *encoded, dones=dones)
                collection_time = timer.end()

            timer.start()
            train_loader, val_loader = self.alg.get_data_loaders(batch_size)
            train_num, val_num = self.alg.get_data_lens()

            for i in range(repeat_learning_times):
                train_loss = 0.
                for p_obs, r_obs, t_encoded, t_action in train_loader:
                    self.optim.zero_grad()
                    s_encoded, s_action = self.student.get_encoded(p_obs, r_obs)
                    loss = self.criterion(s_encoded, t_encoded) + self.criterion(s_action, t_action)
                    train_loss += loss.item()
                    loss.backward()
                    self.optim.step()
                train_loss /= train_num
                log_info(f'Epoch {it:>4} {i}/{repeat_learning_times}  train loss {train_loss:.6f}')

            val_loss = 0.
            with torch.inference_mode():
                for p_obs, r_obs, t_encoded, t_action in val_loader:
                    s_encoded, s_action = self.student.get_encoded(p_obs, r_obs)
                    loss = self.criterion(s_encoded, t_encoded) + self.criterion(s_action, t_action)
                    val_loss += loss.item()
            learning_time = timer.end()
            val_loss /= val_num
            self.log(it, locals(), record_infos)
            if it % save_interval == 0:
                self.save(os.path.join(log_dir, f'model_{it}.pt'))

    def log(self, it, locs, record_infos, width=25):
        log_info(f"{'#' * width}")
        log_info(f"Iteration {it}/{locs['total_iter']}")
        log_info(f"Collection Time: {locs['collection_time']:.3f}")
        log_info(f"Learning Time: {locs['learning_time']:.3f}")

        fps = int(g_cfg.storage_len * self.env.num_envs / (locs['collection_time'] + locs['learning_time']))
        logs = {'Loss/train_loss': locs['train_loss'],
                'Loss/val_loss': locs['val_loss'],
                'Perform/total_fps': fps,
                'Perform/collection time': locs['collection_time'],
                'Perform/learning_time': locs['learning_time']}
        log_info(f"Total Frames: {it * g_cfg.num_envs * g_cfg.storage_len}")
        if record_infos:
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
        wandb.log(logs, step=it)

    def save(self, path, infos=None):
        if not os.path.exists(d := os.path.dirname(path)):
            os.makedirs(d)
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': self.current_epoch,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.student.load_state_dict(loaded_dict['student_state_dict'])
        if load_optimizer:
            self.optim.load_state_dict(loaded_dict['optim_state_dict'])
        self.current_epoch = loaded_dict['epoch']
        return loaded_dict['infos']
