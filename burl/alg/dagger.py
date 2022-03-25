import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from burl.utils import log_info


class Trajectory(Dataset):
    def __init__(self, obs_dim, action_dim, max_size, history_len, device):
        self.obs_dim, self.action_dim, self.max_size = obs_dim, action_dim, max_size
        self.history_len = history_len
        self.obs_buffer = torch.zeros(obs_dim, max_size, device=device)
        self.action_buffer = torch.zeros(action_dim, max_size, device=device)
        self.num_transitions = 0

    def add_transition(self, obs: torch.Tensor, action: torch.Tensor):
        if self.num_transitions >= self.max_size:
            raise RuntimeError('Trajectory overflow')
        self.obs_buffer[:, self.num_transitions] = obs.detach()
        self.action_buffer[:, self.num_transitions] = action.detach()
        self.num_transitions += 1

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        obs_history = self.obs_buffer[:, idx - self.history_len:idx]
        if (padding := self.history_len - idx - 1) > 0:
            obs_history = f.pad(obs_history, (padding, 0))
        action = self.action_buffer[:, idx]
        return obs_history, action


class Dagger(object):
    VAL_RATIO = 0.2

    def __init__(self, num_envs, obs_dim, action_dim, max_size, history_len, device):
        self.init_traj = lambda: Trajectory(obs_dim, action_dim, max_size, history_len, device)
        self.trajectories = [self.init_traj() for _ in range(num_envs)]
        self.train_dataset, self.val_dataset = None, None
        self.train_num, self.val_num = 0, 0
        self.train_val_ratio = (1 - self.VAL_RATIO) / self.VAL_RATIO
        self.num_epochs = 0

    def get_policy(self, teacher, student):
        teacher_prop = self.get_teacher_prop(self.num_epochs)

        def _policy(teacher_obs, student_obs):
            if teacher_prop == 1.0:
                return teacher(teacher_obs)
            elif (student_prop := 1 - teacher_prop) == 1.0:
                return student(student_obs)
            else:
                return teacher(teacher_obs) * teacher_prop + student(student_obs) * student_prop

        return _policy

    def add_transitions(self, observations: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor):
        for idx, traj, obs, action, done in enumerate(zip(self.trajectories, observations, actions, dones)):
            traj.add_transition(obs, action)
            if done:
                self.insert_to_dataset(traj)
                self.trajectories[idx] = self.init_traj()

    def insert_to_dataset(self, traj: Trajectory):
        if not self.train_num or (self.val_num and self.train_num / self.val_num > self.train_val_ratio):
            self.train_dataset = ConcatDataset((self.train_dataset, traj))  # TODO: NOTICE GPU MEM
            self.train_num += len(traj)
        else:
            self.val_dataset = ConcatDataset((self.val_dataset, traj))
            self.val_num += len(traj)

    @staticmethod
    def get_teacher_prop(epoch: int):
        return 1. if epoch == 0 else 0.

    def learn(self, student, optim, criterion, batch_size, repeat_times):
        train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size, shuffle=True)
        for i in range(repeat_times):
            train_loss = 0.
            for obs, des_action in train_loader:
                optim.zero_grad()
                loss = criterion(student(obs), des_action)
                train_loss += loss.item()
                loss.backward()
                optim.step()
            train_loss /= self.train_num
            log_info(f'Epoch {self.num_epochs:>4} {i}/{repeat_times}  train loss {train_loss:.6f}')

        val_loss = 0.
        for obs, des_action in val_loader:
            loss = criterion(student(obs), des_action)
            val_loss += loss.item()
        val_loss /= self.val_num
