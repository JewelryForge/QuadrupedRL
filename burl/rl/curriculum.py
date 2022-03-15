import collections
import math
import multiprocessing as mp
import random
from typing import Union

import numpy as np

from burl.utils import g_cfg


class GameInspiredCurriculum(object):
    def __init__(self, max_difficulty, patience, aggressive=False):
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
            self.decreaseLevel()
            return True
        elif self.combo and self.combo % self.patience == 0:
            self.increaseLevel()
            return True
        return False

    def decreaseLevel(self):
        if self.difficulty > 0:
            self.difficulty -= 1

    def increaseLevel(self):
        if self.difficulty < self.max_difficulty:
            self.difficulty += 1

    def onInit(self, task, robot, env):
        pass

    def onSimulationStep(self, task, robot, env):
        pass

    def onStep(self, task, robot, env):
        pass

    def onReset(self, task, robot, env):
        pass

    def maxLevel(self):
        self.difficulty = self.max_difficulty

    def makeDistribution(self):
        return self


class TerrainCurriculum(GameInspiredCurriculum):
    def __init__(self, aggressive=False):
        super().__init__(100, 1, aggressive)
        self.max_roughness = 0.2
        self.terrain = None
        self.episode_linear_reward_sum = 0.
        self.episode_sim_count = 0

    def generateTerrain(self, sim_env):
        if not self.terrain:
            from burl.sim import Hills
            self.terrain = Hills(size=30, downsample=20, resolution=0.1,
                                 roughness=self.difficulty_degree * self.max_roughness)

            self.terrain.spawn(sim_env)
        else:
            if self.difficulty:
                roughness = self.max_roughness * (random.random() if self.difficulty == self.max_difficulty
                                                  else self.difficulty_degree)
                self.terrain.replaceHeightField(
                    sim_env, self.terrain.makeHeightField(size=30, downsample=20, resolution=0.1, roughness=roughness))
        return self.terrain

    def onSimulationStep(self, task, robot, env):
        self.episode_linear_reward_sum += task.getRewardDetails()['LinearVelocityReward']
        self.episode_sim_count += 1

    def onReset(self, task, robot, env):
        self.register(not env.is_failed and self.episode_linear_reward_sum / self.episode_sim_count > 0.6)
        self.generateTerrain(env.client)
        self.episode_linear_reward_sum = self.episode_sim_count = 0


class DisturbanceCurriculum(GameInspiredCurriculum):
    def __init__(self, aggressive=False):
        super().__init__(50, 1, aggressive)
        self.force_magnitude = np.array(g_cfg.force_magnitude)
        self.torque_magnitude = np.array(g_cfg.torque_magnitude)
        self.interval_range = (500, 1000)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0

    def updateDisturbance(self, env):
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

    def onInit(self, task, robot, env):
        self.updateDisturbance(env)

    def onReset(self, task, robot, env):
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        self.register(not env.is_failed)
        self.updateDisturbance(env)

    def onSimulationStep(self, task, robot, env):
        if env.sim_step >= self.last_update + self.update_interval:
            self.updateDisturbance(env)
            self.update_interval = random.uniform(*self.interval_range)
            self.last_update = env.sim_step


class CurriculumDistribution(object):
    def __init__(self, comm: mp.Queue, difficulty_getter, max_difficulty):
        self.comm, self.difficulty_getter = comm, difficulty_getter
        self.max_difficulty = max_difficulty
        self.difficulty = difficulty_getter()

    @property
    def difficulty_degree(self):
        return self.difficulty / self.max_difficulty

    def onInit(self, task, robot, env):
        pass

    def onSimulationStep(self, task, robot, env):
        pass

    def onStep(self, task, robot, env):
        pass

    def isSuccess(self, task, robot, env) -> bool:
        pass

    def onReset(self, task, robot, env):
        self.comm.put((self.difficulty, self.isSuccess(task, robot, env)))
        self.difficulty = self.difficulty_getter()


class CentralizedCurriculum(object):
    def __init__(self, distribution_class, buffer_len, max_difficulty, bounds=(0.5, 0.8), aggressive=False):
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
        self.buffer.append(success if difficulty == self.difficulty else success * self.upper_bound)
        if len(self.buffer) == self.buffer_len:
            if (mean := sum(self.buffer) / self.buffer_len) > self.upper_bound:
                self.increaseLevel()
            elif mean < self.lower_bound:
                self.decreaseLevel()

    def checkLetter(self):
        while not self.letter_box.empty():
            self.register(*self.letter_box.get())

    def decreaseLevel(self):
        if (difficulty := self.difficulty) > 0:
            self.difficulty = difficulty - 1
            self.buffer.clear()

    def increaseLevel(self):
        if (difficulty := self.difficulty) < self.max_difficulty:
            self.difficulty = difficulty + 1
            self.buffer.clear()

    def makeDistribution(self):
        return self.distribution(self.letter_box, lambda: self._difficulty.value, self.max_difficulty)


class DisturbanceCurriculumDistribution(CurriculumDistribution):
    def __init__(self, comm: mp.Queue, difficulty_getter, max_difficulty):
        super().__init__(comm, difficulty_getter, max_difficulty)
        self.force_magnitude = np.array(g_cfg.force_magnitude)
        self.torque_magnitude = np.array(g_cfg.torque_magnitude)
        self.interval_range = (500, 1000)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0

    updateDisturbance = DisturbanceCurriculum.updateDisturbance

    def onInit(self, task, robot, env):
        self.updateDisturbance(env)

    def isSuccess(self, task, robot, env) -> bool:
        return not env.is_failed

    def onReset(self, task, robot, env):
        super().onReset(task, robot, env)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        self.updateDisturbance(env)

    def onSimulationStep(self, task, robot, env):
        if env.sim_step >= self.last_update + self.update_interval:
            self.updateDisturbance(env)
            self.update_interval = random.uniform(*self.interval_range)
            self.last_update = env.sim_step


class TerrainCurriculumDistribution(CurriculumDistribution):
    def __init__(self, comm: mp.Queue, difficulty_getter, max_difficulty):
        super().__init__(comm, difficulty_getter, max_difficulty)
        self.max_roughness = 0.2
        self.terrain = None
        self.episode_linear_reward_sum = 0.
        self.episode_sim_count = 0

    generateTerrain = TerrainCurriculum.generateTerrain

    def onSimulationStep(self, task, robot, env):
        TerrainCurriculum.onSimulationStep(self, task, robot, env)

    def isSuccess(self, task, robot, env) -> bool:
        return not env.is_failed and self.episode_linear_reward_sum / self.episode_sim_count > 0.6

    def onReset(self, task, robot, env):
        super().onReset(task, robot, env)
        self.generateTerrain(env.client)
        self.episode_linear_reward_sum = self.episode_sim_count = 0


class CentralizedDisturbanceCurriculum(CentralizedCurriculum):
    def __init__(self, buffer_len=32, max_difficulty=25, bounds=(0.5, 0.8), aggressive=False):
        super().__init__(DisturbanceCurriculumDistribution, buffer_len, max_difficulty, bounds, aggressive)


class CentralizedTerrainCurriculum(CentralizedCurriculum):
    def __init__(self, buffer_len=32, max_difficulty=100, bounds=(0.5, 0.8), aggressive=False):
        super().__init__(TerrainCurriculumDistribution, buffer_len, max_difficulty, bounds, aggressive)


CURRICULUM_PROTOTYPE = Union[GameInspiredCurriculum, CentralizedCurriculum]
CURRICULUM_DISTRIB = Union[GameInspiredCurriculum, CurriculumDistribution]
