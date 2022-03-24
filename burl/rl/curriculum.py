import collections
import math
import multiprocessing as mp
import random
from collections.abc import Callable
from typing import Union, Type

import numpy as np

from burl.sim.plugins import Plugin
from burl.utils import g_cfg


class GameInspiredCurriculum(Plugin):
    """A curriculum prototype, with different difficulty between multi environments"""

    def __init__(self, max_difficulty: int, patience: int, aggressive=False):
        self.episode_count = 0
        self.max_difficulty = max_difficulty
        self.patience = patience
        self.difficulty = max_difficulty if aggressive else 0
        self.combo, self.miss = 0, 0

    @property
    def difficulty_degree(self):
        return self.difficulty / self.max_difficulty

    def register(self, success):
        if self.difficulty == self.max_difficulty:
            return False
        self.episode_count += 1
        if success:
            self.miss = 0
            self.combo += 1
        else:
            self.combo = 0
            self.miss += 1
        if self.miss and self.miss % self.patience == 0:
            self.decrease_level()
            return True
        elif self.combo and self.combo % self.patience == 0:
            self.increase_level()
            return True
        return False

    def decrease_level(self):
        if self.difficulty > 0:
            self.difficulty -= 1

    def increase_level(self):
        if self.difficulty < self.max_difficulty:
            self.difficulty += 1

    def on_step(self, task, robot, env):
        return {self.__class__.__name__: self.difficulty_degree}

    def set_max_level(self):
        self.difficulty = self.max_difficulty

    def make_distribution(self):
        """For interface consistence"""
        return self


class TerrainCurriculum(GameInspiredCurriculum):
    utils = ['generate_terrain']

    def __init__(self, aggressive=False):
        super().__init__(100, 1, aggressive)
        self.max_roughness = 0.4
        self.terrain = None
        self.episode_linear_reward_sum = 0.
        self.episode_sim_count = 0

    def generate_terrain(self, sim_env):
        """
        If no terrain has been spawned, create and spawn it.
        Otherwise, update its height field.
        """
        size, resolution = 30, 0.1
        mini_rfn = random.uniform(0, 0.02)
        roughness = self.max_roughness * (random.random() if self.difficulty == self.max_difficulty
                                          else self.difficulty_degree)
        # roughness = self.max_roughness
        if not self.terrain:
            from burl.sim.terrain import Hills
            self.terrain = Hills.make(size, resolution, (roughness, 20), (mini_rfn, 2))
            self.terrain.spawn(sim_env)
        else:
            if self.difficulty:
                self.terrain.replace_heightfield(
                    sim_env, self.terrain.make_heightfield(size, resolution, (roughness, 20), (mini_rfn, 2)))
        return self.terrain

    def on_sim_step(self, task, robot, env):
        self.episode_linear_reward_sum += task.reward_details['LinearVelocityReward']
        self.episode_sim_count += 1

    def on_reset(self, task, robot, env):
        if self.episode_sim_count:
            self.register(not env.is_failed and self.episode_linear_reward_sum / self.episode_sim_count > 0.6)
        self.generate_terrain(env.client)
        self.episode_linear_reward_sum = self.episode_sim_count = 0


class DisturbanceCurriculum(GameInspiredCurriculum):
    def __init__(self, aggressive=False):
        super().__init__(50, 1, aggressive)
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

        # external_torque = (0., 0., 0.)
        env.setDisturbance(external_force, external_torque)

    def on_init(self, task, robot, env):
        self.update_disturbance(env)

    def on_reset(self, task, robot, env):
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        self.register(not env.is_failed)
        self.update_disturbance(env)

    def on_sim_step(self, task, robot, env):
        if env.sim_step >= self.last_update + self.update_interval:
            self.update_disturbance(env)
            self.update_interval = random.uniform(*self.interval_range)
            self.last_update = env.sim_step


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

    def __init__(self, distribution_class: Type[CurriculumDistribution], buffer_len: int, max_difficulty: int,
                 bounds=(0.5, 0.8), aggressive=False):
        self.max_difficulty, self.buffer_len = max_difficulty, buffer_len
        self.distribution = distribution_class
        self.lower_bound, self.upper_bound = bounds
        self.buffer = collections.deque(maxlen=buffer_len)
        self.letter_box = mp.Queue()
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

    def check_letter_box(self):
        while not self.letter_box.empty():
            self.register(*self.letter_box.get())

    def decrease_level(self):
        if (difficulty := self.difficulty) > 0:
            self.difficulty = difficulty - 1
            self.buffer.clear()

    def increase_level(self):
        if (difficulty := self.difficulty) < self.max_difficulty:
            self.difficulty = difficulty + 1
            self.buffer.clear()

    def make_distribution(self) -> Plugin:
        return self.distribution(self.letter_box, lambda: self._difficulty.value, self.max_difficulty)


class DisturbanceCurriculumDistribution(CurriculumDistribution):
    def __init__(self, comm: mp.Queue, difficulty_getter, max_difficulty):
        super().__init__(comm, difficulty_getter, max_difficulty)
        self.force_magnitude = np.array(g_cfg.force_magnitude)
        self.torque_magnitude = np.array(g_cfg.torque_magnitude)
        self.interval_range = (500, 1000)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0

    update_disturbance = DisturbanceCurriculum.update_disturbance

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
        self.max_roughness = 0.2
        self.terrain = None
        self.episode_linear_reward_sum = 0.
        self.episode_sim_count = 0
        self.linear_reward_threshold = 0.4

    generate_terrain = TerrainCurriculum.generate_terrain

    def on_sim_step(self, task, robot, env):
        TerrainCurriculum.on_sim_step(self, task, robot, env)

    def is_success(self, task, robot, env) -> bool:
        average_linear_reward = self.episode_linear_reward_sum / self.episode_sim_count
        return not env.is_failed and average_linear_reward > self.linear_reward_threshold

    def on_reset(self, task, robot, env):
        super().on_reset(task, robot, env)
        self.generate_terrain(env.client)
        self.episode_linear_reward_sum = self.episode_sim_count = 0


class CentralizedDisturbanceCurriculum(CentralizedCurriculum):
    def __init__(self, buffer_len=32, max_difficulty=25, bounds=(0.5, 0.9), aggressive=False):
        super().__init__(DisturbanceCurriculumDistribution, buffer_len, max_difficulty, bounds, aggressive)


class CentralizedTerrainCurriculum(CentralizedCurriculum):
    def __init__(self, buffer_len=32, max_difficulty=50, bounds=(0.5, 0.9), aggressive=False):
        super().__init__(TerrainCurriculumDistribution, buffer_len, max_difficulty, bounds, aggressive)


CURRICULUM_PROTOTYPE = Union[GameInspiredCurriculum, CentralizedCurriculum]
CURRICULUM_DISTRIB = Union[GameInspiredCurriculum, CurriculumDistribution]
