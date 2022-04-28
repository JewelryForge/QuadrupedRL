import collections
import math
import multiprocessing as mp
import random
from collections.abc import Callable
from typing import Union, Type

import numpy as np

from burl.sim.plugins import Plugin
from burl.utils import g_cfg


# class GameInspiredCurriculum(Plugin):
#     """A curriculum prototype, with different difficulty between multi environments"""
#
#     def __init__(self, max_difficulty: int, patience: int, aggressive=False):
#         self.episode_count = 0
#         self.max_difficulty = max_difficulty
#         self.patience = patience
#         self.difficulty = max_difficulty if aggressive else 0
#         self.combo, self.miss = 0, 0
#
#     @property
#     def difficulty_degree(self):
#         return self.difficulty / self.max_difficulty
#
#     def register(self, success):
#         if self.difficulty == self.max_difficulty:
#             return False
#         self.episode_count += 1
#         if success:
#             self.miss = 0
#             self.combo += 1
#         else:
#             self.combo = 0
#             self.miss += 1
#         if self.miss and self.miss % self.patience == 0:
#             self.decrease_level()
#             return True
#         elif self.combo and self.combo % self.patience == 0:
#             self.increase_level()
#             return True
#         return False
#
#     def decrease_level(self):
#         if self.difficulty > 0:
#             self.difficulty -= 1
#
#     def increase_level(self):
#         if self.difficulty < self.max_difficulty:
#             self.difficulty += 1
#
#     def on_step(self, task, robot, env):
#         return {self.__class__.__name__: self.difficulty_degree}
#
#     def set_max_level(self):
#         self.difficulty = self.max_difficulty
#
#     def make_distribution(self):
#         """For interface consistence"""
#         return self
#
#
# class TerrainCurriculum(GameInspiredCurriculum):
#     utils = ['generate_terrain']
#
#     def __init__(self, max_roughness=0.4, aggressive=False):
#         super().__init__(40, 1, aggressive)
#         self.max_roughness = max_roughness
#         self.terrain = None
#         self.episode_linear_reward = 0.
#         self.episode_sim_count = 0
#
#     def generate_terrain(self, sim_env):
#         """
#         If no terrain has been spawned, create and spawn it.
#         Otherwise, update its height field.
#         """
#         size, resolution = 30, 0.1
#         is_init = not self.terrain
#         if is_init:
#             from burl.sim.terrain import Hills
#             # Must initialize terrain with max roughness,
#             # otherwise may cause the robot to get stuck in the terrain.
#             # See https://github.com/bulletphysics/bullet3/issues/4236
#             hills_heightfield = Hills.make_heightfield(size, resolution, (self.max_roughness, 20))
#             hills_heightfield.data *= 1.05
#             self.terrain = Hills(hills_heightfield)
#             self.terrain.spawn(sim_env)
#         if is_init or self.difficulty:
#             if self.difficulty == self.max_difficulty:
#                 roughness = self.max_roughness * self.difficulty_degree
#                 # roughness = self.max_roughness * random.random()
#             else:
#                 roughness = self.max_roughness * self.difficulty_degree
#             heightfield = self.terrain.make_heightfield(size, resolution, (roughness, 20))
#             self.terrain.replace_heightfield(sim_env, heightfield)
#         return self.terrain
#
#     def on_sim_step(self, task, robot, env):
#         self.episode_linear_reward += task.reward_details['UnifiedLinearReward']
#         self.episode_sim_count += 1
#
#     def on_reset(self, task, robot, env):
#         is_success = not env.is_failed and self.episode_linear_reward / self.episode_sim_count > 0.6
#         if self.episode_sim_count:
#             self.register(is_success)
#         self.generate_terrain(env.client)
#         self.episode_linear_reward = 0.
#         self.episode_sim_count = 0
#
#
# class DisturbanceCurriculum(GameInspiredCurriculum):
#     def __init__(self, aggressive=False):
#         super().__init__(50, 1, aggressive)
#         self.force_magnitude = np.array(g_cfg.force_magnitude)
#         self.torque_magnitude = np.array(g_cfg.torque_magnitude)
#         self.interval_range = (500, 1000)
#         self.update_interval = random.uniform(*self.interval_range)
#         self.last_update = 0
#
#     def update_disturbance(self, env):
#         if not self.difficulty:
#             external_force = external_torque = (0., 0., 0.)
#         else:
#             if self.difficulty < self.max_difficulty:
#                 last_difficulty_degree = (self.difficulty - 1) / self.max_difficulty
#                 horizontal_force = np.random.uniform(self.force_magnitude[0] * last_difficulty_degree,
#                                                      self.force_magnitude[0] * self.difficulty_degree)
#                 vertical_force = np.random.uniform(self.force_magnitude[1] * last_difficulty_degree,
#                                                    self.force_magnitude[1] * self.difficulty_degree)
#                 external_torque = np.random.uniform(self.torque_magnitude * last_difficulty_degree,
#                                                     self.torque_magnitude * self.difficulty_degree)
#                 external_torque *= np.random.choice((1, -1), 3)
#             else:
#                 # avoid catastrophic forgetting
#                 horizontal_force = np.random.uniform(0, self.force_magnitude[0])
#                 vertical_force = np.random.uniform(0, self.force_magnitude[1])
#                 external_torque = np.random.uniform(-self.torque_magnitude, self.torque_magnitude)
#
#             yaw = np.random.uniform(0, 2 * math.pi)
#             external_force = np.array((
#                 horizontal_force * np.cos(yaw),
#                 horizontal_force * np.sin(yaw),
#                 vertical_force * np.random.choice((-1, 1))
#             ))
#
#         env.setDisturbance(external_force, external_torque)
#
#     def on_init(self, task, robot, env):
#         self.update_disturbance(env)
#
#     def on_reset(self, task, robot, env):
#         self.update_interval = random.uniform(*self.interval_range)
#         self.last_update = 0
#         self.register(not env.is_failed)
#         self.update_disturbance(env)
#
#     def on_sim_step(self, task, robot, env):
#         if env.sim_step >= self.last_update + self.update_interval:
#             self.update_disturbance(env)
#             self.update_interval = random.uniform(*self.interval_range)
#             self.last_update = env.sim_step


class CurriculumDistribution(Plugin):
    """Defines how curriculum affects the environment"""

    def __init__(self, comm: mp.Queue, difficulty_getter: Callable[[], int], max_difficulty: int):
        self.comm, self.difficulty_getter = comm, difficulty_getter
        self.max_difficulty = max_difficulty
        self.difficulty = difficulty_getter()

    @property
    def difficulty_degree(self):
        return self.difficulty / self.max_difficulty

    def on_step(self, task, robot, env):
        return {self.__class__.__name__: self.difficulty_degree}

    def is_success(self, task, robot, env) -> bool:
        pass

    def on_reset(self, task, robot, env):
        self.comm.put((self.difficulty, self.is_success(task, robot, env)))
        self.difficulty = self.difficulty_getter()


class CentralizedCurriculum(object):
    """
    A centralized curriculum prototype, with common difficulty between multiprocess environments,
    maintains a filter and difficulty alteration.
    """

    def __init__(self, distribution_class: Type[CurriculumDistribution],
                 buffer_len: int, max_difficulty: int,
                 bounds=(0.5, 0.8), aggressive=False):
        self.max_difficulty, self.buffer_len = max_difficulty, buffer_len
        self.distribution = distribution_class
        self.lower_bound, self.upper_bound = bounds
        self.buffer = collections.deque(maxlen=buffer_len)
        self.distrib_infos = mp.Queue()
        self._difficulty = mp.Value('i', max_difficulty if aggressive else 0)

    @property
    def difficulty(self):
        return self._difficulty.value

    @difficulty.setter
    def difficulty(self, value):
        self._difficulty.value = value

    def register(self, difficulty, success: bool):
        self_difficulty = self.difficulty
        self.buffer.append(success if difficulty >= self_difficulty else success * self.upper_bound)
        if len(self.buffer) == self.buffer_len:
            if self_difficulty < self.max_difficulty:
                if (mean := sum(self.buffer) / self.buffer_len) > self.upper_bound:
                    self.increase_level()
                elif mean < self.lower_bound:
                    self.decrease_level()

    def summarize(self):
        while not self.distrib_infos.empty():
            self.register(*self.distrib_infos.get())

    def decrease_level(self):
        if (difficulty := self.difficulty) > 0:
            self.difficulty = difficulty - 1
            self.buffer.clear()

    def increase_level(self):
        if (difficulty := self.difficulty) < self.max_difficulty:
            self.difficulty = difficulty + 1
            self.buffer.clear()

    def value_getter(self):
        return self._difficulty.value

    def make_distribution(self) -> Plugin:
        return self.distribution(self.distrib_infos, self.value_getter, self.max_difficulty)


class DisturbanceCurriculumDistribution(CurriculumDistribution):
    def __init__(self, comm: mp.Queue, difficulty_getter, max_difficulty):
        super().__init__(comm, difficulty_getter, max_difficulty)
        self.force_magnitude = np.array(g_cfg.force_magnitude)
        self.torque_magnitude = np.array(g_cfg.torque_magnitude)
        self.interval_range = (500, 1000)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0

    def update_disturbance(self, env):
        if not self.difficulty:
            external_force = external_torque = (0., 0., 0.)
        else:
            if self.difficulty < self.max_difficulty:
                last_difficulty_degree = (self.difficulty - 1) / self.max_difficulty
                horizontal_force = np.random.uniform(self.force_magnitude[0] * last_difficulty_degree,
                                                     self.force_magnitude[0] * self.difficulty_degree)
                vertical_force = np.random.uniform(self.force_magnitude[1] * last_difficulty_degree,
                                                   self.force_magnitude[1] * self.difficulty_degree)
                external_torque = np.random.uniform(self.torque_magnitude * last_difficulty_degree,
                                                    self.torque_magnitude * self.difficulty_degree)
                external_torque *= np.random.choice((1, -1), 3)
            else:
                # avoid catastrophic forgetting
                horizontal_force = np.random.uniform(0, self.force_magnitude[0])
                vertical_force = np.random.uniform(0, self.force_magnitude[1])
                external_torque = np.random.uniform(-self.torque_magnitude, self.torque_magnitude)

            yaw = np.random.uniform(0, 2 * math.pi)
            external_force = np.array((
                horizontal_force * np.cos(yaw),
                horizontal_force * np.sin(yaw),
                vertical_force * np.random.choice((-1, 1))
            ))
        env.setDisturbance(external_force, external_torque)

    def on_init(self, task, robot, env):
        self.update_disturbance(env)

    def is_success(self, task, robot, env) -> bool:
        return not env.is_failed

    def on_reset(self, task, robot, env):
        super().on_reset(task, robot, env)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        self.update_disturbance(env)

    def on_sim_step(self, task, robot, env):
        if env.sim_step >= self.last_update + self.update_interval:
            self.update_disturbance(env)
            self.update_interval = random.uniform(*self.interval_range)
            self.last_update = env.sim_step


class TerrainCurriculumDistribution(CurriculumDistribution):
    utils = ['generate_terrain']

    def __init__(self, comm: mp.Queue, difficulty_getter, max_difficulty):
        super().__init__(comm, difficulty_getter, max_difficulty)
        self.max_roughness = 0.4
        self.max_slope = 15 / 180 * math.pi
        self.max_step_height = 0.2
        self.terrain = None
        self.episode_linear_reward = 0.
        self.episode_sim_count = 0

    def generate_terrain(self, sim_env):
        """
        If no terrain has been spawned, create and spawn it.
        Otherwise, update its height field.
        """
        from burl.sim.terrain import Hills, Slopes, Steps
        size, resolution = 30, 0.1
        if not self.terrain:
            # Currently, in the master branch of bullet3
            # the robot may get stuck in the terrain.
            # See https://github.com/bulletphysics/bullet3/issues/4236
            # See https://github.com/bulletphysics/bullet3/pull/4253
            self.terrain = Hills.make(size, resolution, (self.max_roughness * self.difficulty_degree, 20))
            self.terrain.spawn(sim_env)
        else:
            terrain_type = random.choice(('hills', 'slopes', 'steps'))
            # terrain_type = 'hills'
            difficulty_degree = random.random() if self.difficulty == self.max_difficulty else self.difficulty_degree
            obj_id, shape_id = self.terrain.id, self.terrain.shape_id
            if terrain_type == 'hills':
                roughness = self.max_roughness * difficulty_degree
                self.terrain = Hills.make(size, resolution, (roughness, 20))
            elif terrain_type == 'slopes':
                slope = self.max_slope * difficulty_degree
                axis = random.choice(('x', 'y'))
                self.terrain = Slopes.make(size, resolution, slope, 3., axis)
            elif terrain_type == 'steps':
                step_height = self.max_step_height * difficulty_degree
                self.terrain = Steps.make(size, resolution, 1., step_height)
            self.terrain.terrain_id, self.terrain.terrain_shape_id = obj_id, shape_id
            self.terrain.replace_heightfield(sim_env)
        return self.terrain

    def on_sim_step(self, task, robot, env):
        # self.episode_linear_reward += task.reward_details['UnifiedLinearReward']
        self.episode_sim_count += 1

    def is_success(self, task, robot, env) -> bool:
        average_linear_reward = self.episode_linear_reward / self.episode_sim_count
        return not env.is_failed and average_linear_reward > 0.6

    def on_reset(self, task, robot, env):
        super().on_reset(task, robot, env)
        env.terrain = self.generate_terrain(env.client)
        self.episode_linear_reward = self.episode_sim_count = 0


class CentralizedDisturbanceCurriculum(CentralizedCurriculum):
    def __init__(self, buffer_len=32, max_difficulty=10, bounds=(0.5, 0.8), aggressive=False):
        super().__init__(DisturbanceCurriculumDistribution, buffer_len, max_difficulty, bounds, aggressive)


class CentralizedTerrainCurriculum(CentralizedCurriculum):
    def __init__(self, buffer_len=32, max_difficulty=20, bounds=(0.5, 0.8), aggressive=False):
        super().__init__(TerrainCurriculumDistribution, buffer_len, max_difficulty, bounds, aggressive)


# CURRICULUM_PROTOTYPE = Union[GameInspiredCurriculum, CentralizedCurriculum]
# CURRICULUM_DISTRIB = Union[GameInspiredCurriculum, CurriculumDistribution]

CURRICULUM_PROTOTYPE = Union[CentralizedCurriculum]
CURRICULUM_DISTRIB = Union[CurriculumDistribution]
