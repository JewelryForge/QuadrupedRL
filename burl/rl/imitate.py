import os
from collections import deque

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm

from burl.alg.ac import Actor
from burl.alg.dagger import Dagger
from burl.alg.student import Student
from burl.rl.runner import Accountant
from burl.rl.task import get_task, CentralizedTask
from burl.sim.env import FixedTgEnv, AlienGo
from burl.sim.motor import ActuatorNetManager
from burl.sim.parallel import EnvContainerMp2, EnvContainer
from burl.sim.state import ExteroObservation, RealWorldObservation, Action, ProprioInfo
from burl.utils import g_cfg, to_dev, MfTimer, log_info, make_part


class ImitationLearning(object):
    def __init__(self, num_envs, make_env, teacher, make_alg, make_optim, device):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.env = (EnvContainerMp2 if g_cfg.use_mp else EnvContainer)(make_env, num_envs)
        self.teacher: Actor = teacher.to(self.device)
        self.student: Student = Student(self.teacher).to(self.device)
        self.alg: Dagger = make_alg()
        self.optim = make_optim(self.student.parameters())
        self.current_epoch = 0
        self.criterion = torch.nn.MSELoss()

    def learn(self, num_epochs, num_steps_each_epoch, batch_size, repeat_learning_times,
              save_interval, log_dir, record_infos: bool):
        proprio_obs, realworld_obs, extended_obs = to_dev(*self.env.init_observations())
        self.alg.add_transitions(proprio_obs, realworld_obs, *self.teacher.get_encoded(extended_obs))
        if record_infos:
            accountant = Accountant()
            cur_reward_sum = torch.zeros(self.num_envs, device=self.device)
            cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)
            reward_buffer, eps_len_buffer = deque(maxlen=self.num_envs), deque(maxlen=self.num_envs)
        timer = MfTimer()
        for it in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch += 1
            log_info(f"{'#' * 25}")
            with torch.inference_mode():
                timer.start()
                policy = self.alg.get_policy(self.teacher, self.student)
                for _ in tqdm(range(num_steps_each_epoch)):
                    actions = policy(extended_obs, (self.alg.get_proprio_history(), realworld_obs))
                    (proprio_obs, realworld_obs, extended_obs), rewards, dones, infos = self.env.step(actions)
                    proprio_obs, realworld_obs, extended_obs = to_dev(proprio_obs, realworld_obs, extended_obs)

                    if record_infos:
                        cur_reward_sum += rewards.to(self.device)
                        cur_episode_length += 1
                        accountant.register(infos['reward_details'])

                    if any(dones):
                        reset_ids = torch.squeeze(dones.nonzero(), dim=1)
                        proprio_obs_r, real_world_obs_r, extended_obs_r = to_dev(*self.env.reset(reset_ids))
                        proprio_obs[(reset_ids,)] = proprio_obs_r
                        realworld_obs[(reset_ids,)] = real_world_obs_r
                        extended_obs[(reset_ids,)] = extended_obs_r
                        if record_infos:
                            reward_buffer.extend(cur_reward_sum[reset_ids].cpu().numpy().tolist())
                            eps_len_buffer.extend(cur_episode_length[reset_ids].cpu().numpy().tolist())
                            cur_reward_sum[reset_ids] = 0
                            cur_episode_length[reset_ids] = 0

                    task_infos = infos.get('task_info', {})
                    encoded = self.teacher.get_encoded(extended_obs)
                    self.alg.add_transitions(proprio_obs, realworld_obs, *encoded, dones=dones)

            collection_time = timer.end()
            log_info(f"Collection Time: {collection_time:.3f}")
            timer.start()
            train_loader, val_loader = self.alg.get_data_loaders(batch_size)
            train_num, val_num = self.alg.get_data_lens()
            log_info(f'Dataset Size: train {train_num}, val {val_num}')
            train_losses_list = []
            num_losses = 0

            train_losses = np.zeros(3)
            for i in range(repeat_learning_times):
                for p_obs, r_obs, t_encoded, t_action in tqdm(train_loader):
                    self.optim.zero_grad()
                    s_encoded, s_action = self.student.get_encoded(p_obs, r_obs)
                    encoded_loss = self.criterion(s_encoded, t_encoded)
                    action_loss = self.criterion(s_action, t_action) * 10
                    loss = encoded_loss + action_loss
                    train_losses += (loss.item(), encoded_loss.item(), action_loss.item())
                    num_losses += 1
                    loss.backward()
                    self.optim.step()

            train_losses /= num_losses
            log_info(f'Epoch {it:>4}  encode loss {train_losses[1]:.6f}')
            log_info(f'Epoch {it:>4}  action loss {train_losses[2]:.6f}')

            num_losses = 0
            val_losses = np.zeros(3)
            with torch.inference_mode():
                for p_obs, r_obs, t_encoded, t_action in tqdm(val_loader):
                    s_encoded, s_action = self.student.get_encoded(p_obs, r_obs)
                    encoded_loss = self.criterion(s_encoded, t_encoded)
                    action_loss = self.criterion(s_action, t_action) * 10
                    loss = encoded_loss + action_loss
                    val_losses += (loss.item(), encoded_loss.item(), action_loss.item())
                    num_losses += 1

            val_losses /= num_losses
            log_info(f'Epoch {it:>4}  val encode loss {val_losses[1]:.6f}')
            log_info(f'Epoch {it:>4}  val action loss {val_losses[2]:.6f}')
            learning_time = timer.end()
            log_info(f"Learning Time: {learning_time:.3f}")
            self.log(it, locals(), record_infos)
            if it % save_interval == 0:
                self.save(os.path.join(log_dir, f'model_{it}.pt'))

    def log(self, it, locs, record_infos):
        logs = {'Loss/train/total': locs['train_losses'][0],
                'Loss/train/encode': locs['train_losses'][1],
                'Loss/train/action': locs['train_losses'][2],
                'Loss/val/total': locs['val_losses'][0],
                'Loss/val/encode': locs['val_losses'][1],
                'Loss/val/action': locs['val_losses'][2],
                'Perform/collection time': locs['collection_time'],
                'Perform/learning_time': locs['learning_time']}
        log_info(f"Total Frames: {it * self.num_envs * locs['num_steps_each_epoch']}")
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


class ImitationRunner(object):
    def __init__(self, model_path, task_type='basic'):
        teacher = Actor(ExteroObservation.dim, RealWorldObservation.dim, Action.dim,
                        g_cfg.extero_layer_dims, g_cfg.proprio_layer_dims, g_cfg.action_layer_dims)
        model_info = torch.load(model_path)
        log_info(f'Loading model {model_path}')
        teacher.load_state_dict(model_info['actor_state_dict'])
        task_prototype = CentralizedTask()
        if g_cfg.actuator_net:
            self.acnet_manager = ActuatorNetManager(g_cfg.actuator_net)
        else:
            self.acnet_manager = g_cfg.actuator_net
        make_robot = AlienGo.auto_maker(actuator_net=self.acnet_manager)

        self.obj = ImitationLearning(
            g_cfg.num_envs,
            make_part(FixedTgEnv, make_robot, task_prototype.spawner(get_task(task_type)),
                      obs_types=('noisy_proprio_info', 'noisy_realworld', 'noisy_extended')),
            teacher,
            make_part(Dagger, g_cfg.num_envs, 2000, g_cfg.history_len,
                      ProprioInfo.dim, RealWorldObservation.dim, g_cfg.extero_layer_dims[-1], Action.dim,
                      device=g_cfg.dev),
            make_part(AdamW, lr=g_cfg.learning_rate, weight_decay=1e-2),
            device=g_cfg.dev)

    def learn(self):
        self.obj.learn(g_cfg.num_iterations, g_cfg.num_steps_each_epoch, g_cfg.batch_size, g_cfg.repeat_times,
                       g_cfg.save_interval, g_cfg.log_dir, True)
