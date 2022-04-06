from collections.abc import Iterable
from typing import overload, Callable

import numpy as np
import torch
import torch.multiprocessing as mp

from burl.sim.env import QuadrupedEnv

__all__ = ['EnvContainer', 'EnvContainerMp2', 'SingleEnvContainer']

SINGLE = np.float32


class EnvContainer(object):
    @overload
    def __init__(self, make_env: Callable[..., QuadrupedEnv], num_envs: int):
        ...

    @overload
    def __init__(self, make_env: Iterable[Callable[..., QuadrupedEnv]], num_envs: None):
        ...

    def __init__(self, make_env, num_envs=None):
        self.num_envs = num_envs
        self.make_env = make_env
        self._envs = []

    def step(self, actions: torch.Tensor):
        return self.merge_results([e.step(a) for e, a in zip(self._envs, actions.cpu().numpy())])

    def __del__(self):
        self.close()

    @staticmethod
    def merge_results(results):
        observations, rewards, dones, infos = zip(*results)

        def _merge_dict_recursively(_infos: list[dict]):
            _infos_merged = {}
            for _k, _v in _infos[0].items():
                if isinstance(_v, dict):
                    _infos_merged[_k] = _merge_dict_recursively([_info[_k] for _info in _infos])
                else:
                    _infos_merged[_k] = torch.Tensor(np.array([_info[_k] for _info in _infos]))
            return _infos_merged

        return (tuple(torch.from_numpy(np.array(obs, dtype=SINGLE)) for obs in zip(*observations)),
                torch.from_numpy(np.array(rewards, dtype=SINGLE)),
                torch.from_numpy(np.array(dones, dtype=SINGLE)),
                _merge_dict_recursively(infos))

    def reset(self, ids):
        observations = zip(*[self._envs[i].reset() for i in ids])
        return tuple(torch.from_numpy(np.array(obs, dtype=SINGLE)) for obs in observations)

    def init_observations(self):
        if isinstance(self.make_env, Iterable):
            self._envs: list[QuadrupedEnv] = [make() for make in self.make_env]
            self.num_envs = len(self._envs)
        else:
            self._envs: list[QuadrupedEnv] = [self.make_env() for _ in range(self.num_envs)]
        self._envs[0].task.report()
        observations = zip(*[env.initObservation() for env in self._envs])
        # TO MY ASTONISHMENT, A LIST COMPREHENSION IS FASTER THAN A GENERATOR!!!
        return tuple(torch.from_numpy(np.array(obs, dtype=SINGLE)) for obs in observations)

    def close(self):
        pass


class SingleEnvContainer(EnvContainer):
    def __init__(self, make_env):
        super().__init__(make_env, 1)

    @property
    def unwrapped(self) -> QuadrupedEnv:
        return self._envs[0]


# class EnvContainerMultiProcess(EnvContainer):
#     def __init__(self, make_env, num_envs=None, num_processes=4):
#         super().__init__(make_env, num_envs)
#         self._num_processes = num_processes
#         self._queues = [Queue() for _ in range(num_processes)]
#
#     def step_in_process(self, action, env_id, queue_id):
#         self._queues[queue_id].put(self._envs[env_id].step(action))
#
#     def step(self, actions: torch.Tensor):
#         actions = [Action.from_array(action.cpu().numpy()) for action in actions]
#         results = []
#         for i in range(math.ceil(self.num_envs / self._num_processes)):
#             processes = []
#             remains = min(self.num_envs - i * self._num_processes, self._num_processes)
#             for j in range(remains):
#                 idx = i * self._num_processes + j
#                 p = Process(target=self.step_in_process, args=(actions[idx], idx, j))
#                 processes.append(p)
#                 p.start()
#             for p, q in zip(processes, self._queues):
#                 results.append(q.get())
#                 p.join()
#         return self.merge_results(results)

class torch_single_thread(object):
    def __init__(self):
        self.num_threads = torch.get_num_threads()
        # self.num_interop_threads = torch.get_num_interop_threads()

    def __enter__(self):
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_num_threads(self.num_threads)
        # torch.set_num_interop_threads(self.num_interop_threads)


class EnvContainerMp2(EnvContainer):
    CMD_RESET = 0
    CMD_ACT = 1
    CMD_EXIT = 2

    def __init__(self, make_env, num_envs=None):
        super().__init__(make_env, num_envs)
        self._conn_in_proc, self._conn, self._processes = [], [], []
        for _ in range(self.num_envs):
            conn1, conn2 = mp.Pipe(duplex=True)
            proc = mp.Process(target=self.step_in_process, args=(make_env, conn1))
            proc.start()
            self._conn_in_proc.append(conn1)
            self._conn.append(conn2)
            self._processes.append(proc)

    @staticmethod
    def step_in_process(make_env, conn):
        env = make_env()
        # Explicitly prohibit multithread AcNet calculation
        torch.set_num_threads(1)
        obs = env.initObservation()
        conn.send(obs)
        while True:
            action_type, *content = conn.recv()
            if action_type == EnvContainerMp2.CMD_RESET:
                obs = env.reset()
                conn.send(obs)
            elif action_type == EnvContainerMp2.CMD_ACT:
                obs = env.step(content[0])
                conn.send(obs)
            elif action_type == EnvContainerMp2.CMD_EXIT:
                return
            else:
                raise RuntimeError(f'Unknown action_type {action_type}')

    def step(self, actions: torch.Tensor):
        for action, conn in zip(actions.cpu().numpy(), self._conn):
            conn.send((self.CMD_ACT, action))
        results = [conn.recv() for conn in self._conn]
        return self.merge_results(results)

    def close(self):
        for conn in self._conn:
            conn.send((self.CMD_EXIT,))
        for proc in self._processes:
            proc.join()

    def init_observations(self):
        results = [conn.recv() for conn in self._conn]
        return (torch.Tensor(np.asarray(o)) for o in zip(*results))

    def reset(self, ids):
        for i in ids:
            self._conn[i].send((self.CMD_RESET,))
        observations = zip(*[self._conn[i].recv() for i in ids])
        return tuple([torch.Tensor(np.asarray(obs)) for obs in observations])
