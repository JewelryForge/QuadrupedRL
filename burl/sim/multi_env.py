import logging
import math
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, Pipe
from typing import Tuple, Union

import numpy as np
import torch

from burl.rl.state import ExtendedObservation, Action
from burl.sim.env import TGEnv


class EnvContainer(object):
    num_obs = ExtendedObservation.dim
    num_privileged_obs = ExtendedObservation.dim

    def __init__(self, num_envs, make_env, device='cuda'):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = 1000
        self.extras = {}
        self._num_envs = num_envs
        self._envs: list[TGEnv] = [make_env() for _ in range(num_envs)]

    def step(self, actions: torch.Tensor):
        actions = [Action.from_array(action.cpu().numpy()) for action in actions]
        # logging.debug('ACTION:', *(action.__dict__ for action in actions))
        return self.merge_results([e.step(a) for e, a in zip(self._envs, actions)])

    def __del__(self):
        self.close()

    @staticmethod
    def merge_results(results):
        pri_observations, observations, rewards, dones, infos = zip(*results)
        infos = {k: torch.tensor(np.array([info[k] for info in infos])) for k in infos[0]}
        return (torch.Tensor(np.array(pri_observations)), torch.Tensor(np.array(observations)),
                torch.Tensor(np.array(rewards)), torch.Tensor(np.array(dones)), infos)

    def reset(self, dones):
        for env, done in zip(self._envs, dones):
            if done:
                env.reset()

    def init_observations(self):
        # TO MY ASTONISHMENT, A LIST COMPREHENSION IS FASTER THAN A GENERATOR!!!
        return (torch.Tensor(np.asarray(o)) for o in zip(*[env.initObservation() for env in self._envs]))

    def close(self):
        pass


class EnvContainerMultiProcess(EnvContainer):
    def __init__(self, num_envs, make_env, num_processes=4, device='cuda'):
        super().__init__(num_envs, make_env, device)
        self._num_processes = num_processes
        self._queues = [Queue() for _ in range(num_processes)]

    def step_in_process(self, action, env_id, queue_id):
        self._queues[queue_id].put(self._envs[env_id].step(action))

    def step(self, actions: torch.Tensor):
        actions = [Action.from_array(action.cpu().numpy()) for action in actions]
        results = []
        for i in range(math.ceil(self.num_envs / self._num_processes)):
            processes = []
            remains = min(self.num_envs - i * self._num_processes, self._num_processes)
            for j in range(remains):
                idx = i * self._num_processes + j
                p = Process(target=self.step_in_process, args=(actions[idx], idx, j))
                processes.append(p)
                p.start()
            for p, q in zip(processes, self._queues):
                results.append(q.get())
                p.join()
        return self.merge_results(results)


class EnvContainerMultiProcess2(EnvContainer):
    def __init__(self, num_envs, make_env, device='cuda'):
        super().__init__(num_envs, make_env, device)
        self._num_processes = num_envs
        self._conn1, self._conn2 = zip(*[Pipe(duplex=True) for _ in range(num_envs)])
        self._processes = [Process(target=self.step_in_process, args=(env, conn,))
                           for env, conn in zip(self._envs, self._conn1)]
        for p in self._processes:
            p.start()

    def step_in_process(self, env, conn):
        obs = env.initObservation()
        conn.send(obs)
        while True:
            action = conn.recv()
            if action == 'reset':
                env.reset()
            elif isinstance(action, Action):
                obs = env.step(action)
                conn.send(obs)
            elif action is None:
                return
            else:
                raise RuntimeError(f'Unknown action {action}')

    def step(self, actions: torch.Tensor):
        actions = [Action.from_array(action.cpu().numpy()) for action in actions]
        for action, conn in zip(actions, self._conn2):
            conn.send(action)
        results = [conn.recv() for conn in self._conn2]
        return self.merge_results(results)

    def close(self):
        for conn in self._conn2:
            conn.send(None)
        for proc in self._processes:
            proc.join()

    def init_observations(self):
        results = [conn.recv() for conn in self._conn2]
        return (torch.Tensor(np.asarray(o)) for o in zip(*results))

    def reset(self, dones):
        for conn, done in zip(self._conn2, dones):
            if done:
                conn.send('reset')


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
