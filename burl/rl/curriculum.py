import math
import random

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


class TerrainCurriculum(GameInspiredCurriculum):
    def __init__(self, aggressive=False):
        super().__init__(100, 1, aggressive)
        self.max_roughness = 0.4
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
