import math
from collections import Iterable
from multiprocessing import Process, Queue, Pipe

import numpy as np
import torch

from burl.rl.state import Action
from burl.sim.env import TGEnv


class EnvContainer(object):
    def __init__(self, make_env, num_envs=None):
        self.num_envs = num_envs
        if isinstance(make_env, Iterable):
            self._envs: list[TGEnv] = [make() for make in make_env]
            self.num_envs = len(self._envs)
        else:
            self._envs: list[TGEnv] = [make_env() for _ in range(self.num_envs)]

    def step(self, actions: torch.Tensor):
        actions = [Action.from_array(action.cpu().numpy()) for action in actions]
        return self.merge_results([e.step(a) for e, a in zip(self._envs, actions)])

    def __del__(self):
        self.close()

    @staticmethod
    def merge_results(results):
        pri_observations, observations, rewards, dones, infos = zip(*results)
        infos_merged = {}
        for k in infos[0]:
            if isinstance(infos[0][k], dict):
                info_item = [info[k] for info in infos]
                infos_merged[k] = {k: torch.tensor(np.array([info[k] for info in info_item])) for k in info_item[0]}
            else:
                infos_merged[k] = torch.tensor(np.array([info[k] for info in infos]))
        return (torch.Tensor(np.array(pri_observations)), torch.Tensor(np.array(observations)),
                torch.Tensor(np.array(rewards)), torch.Tensor(np.array(dones)), infos_merged)

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
    def __init__(self, make_env, num_envs=None, num_processes=4):
        super().__init__(make_env, num_envs)
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
    def __init__(self, make_env, num_envs=None):
        super().__init__(make_env, num_envs)
        self._conn1, self._conn2 = zip(*[Pipe(duplex=True) for _ in range(self.num_envs)])
        self._processes = [Process(target=self.step_in_process, args=(env, conn,))
                           for env, conn in zip(self._envs, self._conn1)]
        for p in self._processes:
            p.start()

    @staticmethod
    def step_in_process(env, conn):
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
