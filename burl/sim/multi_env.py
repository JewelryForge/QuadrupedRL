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
        self._envs[0].task.report()

    def step(self, actions: torch.Tensor):
        actions = [Action.from_array(action.cpu().numpy()) for action in actions]
        # print(actions[0].__dict__)
        return self.merge_results([e.step(a) for e, a in zip(self._envs, actions)])

    def __del__(self):
        self.close()

    @staticmethod
    def merge_results(results):
        p_obs, obs, rewards, dones, infos = zip(*results)

        def _merge_dict_recursively(_infos: list[dict]):
            _infos_merged = {}
            for _k, _v in _infos[0].items():
                if isinstance(_v, dict):
                    _infos_merged[_k] = _merge_dict_recursively([_info[_k] for _info in _infos])
                else:
                    _infos_merged[_k] = torch.tensor(np.array([_info[_k] for _info in _infos]))
            return _infos_merged

        return (torch.Tensor(np.array(p_obs)), torch.Tensor(np.array(obs)),
                torch.Tensor(np.array(rewards)), torch.Tensor(np.array(dones)), _merge_dict_recursively(infos))

    def reset(self, ids):
        p_obs, obs = zip(*[self._envs[i].reset() for i in ids])
        return torch.Tensor(np.array(p_obs)), torch.Tensor(np.array(obs))

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
                obs = env.reset()
                conn.send(obs)
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

    def reset(self, ids):
        for i in ids:
            self._conn2[i].send('reset')
        p_obs, obs = zip(*[self._conn2[i].recv() for i in ids])
        return torch.Tensor(np.array(p_obs)), torch.Tensor(np.array(obs))
