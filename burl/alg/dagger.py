from typing import Union

import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class Trajectory(Dataset):
    def __init__(self, max_len, history_len, proprio_obs_dim, *other_obs_dims, device):
        self.max_len, self.history_len = max_len, history_len
        self.proprio_obs_buffer = torch.zeros(proprio_obs_dim, max_len, device=device)
        self.other_buffers = []
        for dim in other_obs_dims:
            self.other_buffers.append(torch.zeros(dim, max_len, device=device))
        self.num_transitions = 0

    def add_transition(self, proprio_obs: torch.Tensor, *other_obs: torch.Tensor):
        if self.num_transitions >= self.max_len:
            raise RuntimeError('Trajectory overflow')
        self.proprio_obs_buffer[:, self.num_transitions] = proprio_obs.detach()
        for obs, buffer in zip(other_obs, self.other_buffers):
            buffer[:, self.num_transitions] = obs.detach()
        self.num_transitions += 1

    def __len__(self):
        return self.num_transitions

    def get_proprio_history(self):
        if (padding := self.history_len - self.num_transitions) > 0:
            proprio_history = f.pad(self.proprio_obs_buffer[:, 0:self.num_transitions], (padding, 0))
        else:
            proprio_history = self.proprio_obs_buffer[:, self.num_transitions - self.history_len:self.num_transitions]
        return proprio_history

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.num_transitions - idx - 2
        padding = self.history_len - idx - 1
        proprio_obs_history = self.proprio_obs_buffer[:, max(-padding, 0): idx + 1]
        if padding > 0:
            proprio_obs_history = f.pad(proprio_obs_history, (padding, 0))
        other_obs = [buffer[:, idx] for buffer in self.other_buffers]
        return proprio_obs_history, *other_obs


class SlidingWindow(object):
    def __init__(self, obs_dim, max_len, history_len, device):
        self.obs_buffer = torch.zeros(1, obs_dim, max_len, device=device)
        self.max_len, self.history_len = max_len, history_len
        self.num_transitions = 0

    def add_transition(self, proprio_obs: torch.Tensor):
        if self.num_transitions >= self.max_len:
            self.num_transitions = self.history_len - 1
            self.obs_buffer[..., :self.num_transitions] = self.obs_buffer[..., -self.num_transitions:]
        self.obs_buffer[..., self.num_transitions] = proprio_obs.detach()
        self.num_transitions += 1

    def get_window(self):
        if (padding := self.history_len - self.num_transitions) > 0:
            obs_history = f.pad(self.obs_buffer[..., 0:self.num_transitions], (padding, 0))
        else:
            obs_history = self.obs_buffer[..., self.num_transitions - self.history_len:self.num_transitions]
        return obs_history

    def clear(self):
        self.num_transitions = 0


class Dagger(object):
    VAL_RATIO = 0.2

    def __init__(self, num_envs, max_len, history_len, proprio_obs_dim, *other_obs_dims, device):
        self.init_traj = lambda: Trajectory(max_len, history_len, proprio_obs_dim, *other_obs_dims, device=device)
        self.trajectories: list[Trajectory] = [self.init_traj() for _ in range(num_envs)]
        self.train_dataset, self.val_dataset = None, None
        self.train_num, self.val_num = 0, 0
        self.train_val_ratio = (1 - self.VAL_RATIO) / self.VAL_RATIO
        self.num_epochs = 0

    def get_policy(self, teacher, student):
        teacher_prop = self.get_teacher_prop(self.num_epochs)

        def _policy(teacher_obs: Union[torch.Tensor, tuple[torch.Tensor]],
                    student_obs: Union[torch.Tensor, tuple[torch.Tensor]]) -> torch.Tensor:
            tp, sp = teacher.get_policy(), student.get_policy()
            if teacher_prop == 1.0:
                return tp(*teacher_obs) if isinstance(teacher_obs, tuple) else tp(teacher_obs)
            elif (student_prop := 1 - teacher_prop) == 1.0:
                return sp(*student_obs) if isinstance(student_obs, tuple) else sp(student_obs)
            else:
                teacher_action = tp(*teacher_obs) if isinstance(teacher_obs, tuple) else tp(teacher_obs)
                student_action = sp(*student_obs) if isinstance(student_obs, tuple) else sp(student_obs)
                return teacher_action * teacher_prop + student_action * student_prop

        return _policy

    def add_transitions(self, proprio_obs: torch.Tensor, *other_obs: torch.Tensor, dones: torch.Tensor = False):
        if dones is not False:
            for idx in torch.nonzero(dones):
                self.insert_into_dataset(self.trajectories[idx])
                self.trajectories[idx] = self.init_traj()
        for idx, (traj, p_obs, *o_obs) in enumerate(
                zip(self.trajectories, proprio_obs, *other_obs)):
            traj.add_transition(p_obs, *o_obs)

    def get_obs(self):
        observations = [traj[-1] for traj in self.trajectories]
        return [torch.stack(obs) for obs in zip(*observations)]

    def get_proprio_history(self):
        return torch.stack([traj.get_proprio_history() for traj in self.trajectories])

    def insert_into_dataset(self, traj: Trajectory):  # TODO: NOTICE GPU MEM
        if not self.train_num or (self.val_num and self.train_num / self.val_num < self.train_val_ratio):
            self.train_dataset = ConcatDataset((self.train_dataset, traj)) if self.train_dataset else traj
            self.train_num += len(traj)
        else:
            self.val_dataset = ConcatDataset((self.val_dataset, traj)) if self.val_dataset else traj
            self.val_num += len(traj)

    @staticmethod
    def get_teacher_prop(epoch: int):
        return 1. if epoch == 0 else 0.

    def get_data_loaders(self, batch_size):
        return (DataLoader(self.train_dataset, batch_size, shuffle=True),
                DataLoader(self.val_dataset, batch_size, shuffle=True))

    def get_data_lens(self):
        return self.train_num, self.val_num
