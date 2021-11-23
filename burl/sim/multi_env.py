import math
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import Tuple, Union

import numpy as np
import torch

from burl.rl.state import ExtendedObservation


class EnvContainer(object):
    num_obs = ExtendedObservation.dim
    num_privileged_obs = ExtendedObservation.dim

    def __init__(self, num_envs, make_env, use_gui=False, device='cuda'):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        self._num_envs = num_envs
        if use_gui:
            self._envs = [make_env(use_gui=True)] + [make_env(use_gui=False) for _ in range(num_envs - 1)]
        else:
            self._envs = [make_env() for _ in range(num_envs)]

    def step(self, actions: torch.Tensor):
        actions = actions.cpu().numpy()
        pri_observations, observations, rewards, dones, _ = zip(*[e.step(a) for e, a in zip(self._envs, actions)])
        self.reset(i for i in range(self.num_envs) if dones[i] is True)
        return (torch.Tensor(np.array(pri_observations)), torch.Tensor(np.array(observations)),
                torch.Tensor(np.array(rewards)), torch.Tensor(np.array(dones)), {})

    def reset(self, env_ids):
        for i in env_ids:
            # print(i, 'reset')
            self._envs[i].reset()

    def init_observations(self):
        # TO MY ASTONISHMENT, A LIST COMPREHENSION IS FASTER THAN A GENERATOR!!!
        return (torch.Tensor(np.asarray(o)) for o in zip(*[env.init_observation() for env in self._envs]))


class EnvContainerMultiProcess(EnvContainer):
    def __init__(self, num_envs, make_env, num_processes=4, device='cuda'):
        super().__init__(num_envs, make_env, False, device)
        self._num_processes = num_processes
        self._queues = [Queue() for _ in range(num_processes)]

    def step_in_process(self, action, env_id, queue_id):
        self._queues[queue_id].put(self._envs[env_id].step(action))

    def step(self, actions: torch.Tensor):
        actions = actions.cpu().numpy()
        results = []
        for i in range(math.ceil(self.num_envs / self._num_processes)):
            processes = []
            remains = min(self.num_envs - i * self._num_processes, self._num_processes)
            for j in range(remains):
                idx = i * self._num_processes + j
                p = Process(self.step_in_process(actions[idx], env_id=idx, queue_id=j))
                processes.append(p)
                p.start()
            for p, q in zip(processes, self._queues):
                p.join()
                results.append(q.get())
        pri_observations, observations, rewards, dones, _ = zip(*results)
        self.reset(i for i in range(self.num_envs) if dones[i] is True)
        return (torch.Tensor(np.array(pri_observations)), torch.Tensor(np.array(observations)),
                torch.Tensor(np.array(rewards)), torch.Tensor(np.array(dones)), {})


class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # current episode duration
    extras: dict
    device: torch.device

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[
        torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass

    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass
