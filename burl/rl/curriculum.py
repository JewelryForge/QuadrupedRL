import math
import random

import numpy as np

from burl.utils import g_cfg, log_debug


class GameInspiredCurriculum(object):
    def __init__(self, max_difficulty, patience, aggressive=False):
        self.episode_counter = 0
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
        self.episode_counter += 1
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

    def onInit(self, cmd, robot, env):
        pass

    def onSimulationStep(self, cmd, robot, env):
        pass

    def onStep(self, cmd, robot, env):
        pass

    def onReset(self, cmd, robot, env):
        pass

    def maxLevel(self):
        self.difficulty = self.max_difficulty


class TerrainCurriculum(GameInspiredCurriculum):
    pass


class DisturbanceCurriculum(GameInspiredCurriculum):
    def __init__(self, aggressive=False):
        super().__init__(10, 5, aggressive)
        self.force_magnitude = np.array(g_cfg.force_magnitude)
        self.torque_magnitude = np.array(g_cfg.torque_magnitude)
        self.interval_range = (500, 1000)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0

    def updateDisturbance(self, env):
        if self.difficulty:
            force_magnitude = self.force_magnitude * self.difficulty_degree
            torque_magnitude = self.torque_magnitude * self.difficulty_degree
            horizontal_force = np.random.uniform(0, force_magnitude[0] * self.difficulty_degree)
            yaw = np.random.uniform(0, 2 * math.pi)
            vertical_force = np.random.uniform(0, force_magnitude[1] * self.difficulty_degree)
            external_force = np.array((
                horizontal_force * np.cos(yaw),
                horizontal_force * np.sin(yaw),
                vertical_force * np.random.choice((-1, 1))
            ))

            external_torque = (0., 0., 0.)
            # external_torque = np.random.uniform(-torque_magnitude, torque_magnitude)
            env.setDisturbance(external_force, external_torque)

    def onInit(self, cmd, robot, env):
        self.updateDisturbance(env)

    def onReset(self, cmd, robot, env):
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        self.register(not env.is_failed)
        self.updateDisturbance(env)

    def onSimulationStep(self, cmd, robot, env):
        if env.sim_step >= self.last_update + self.update_interval:
            self.updateDisturbance(env)
            self.update_interval = random.uniform(*self.interval_range)
            self.last_update = env.sim_step

# class TerrainCurriculum(GameInspiredCurriculum):
#     def __init__(self, bullet_client):
#         super().__init__()
#         self.bullet_client = bullet_client
#         self.terrain = makeStandardRoughTerrain(self.bullet_client, 0.0)
#         self.counter = 0
#         self.difficulty = 0.0
#         self.difficulty_level = 0
#         self.combo, self.miss = 0, 0
#
#     def decrease_level(self):
#         if self.difficulty_level > 0:
#             self.difficulty -= g_cfg.difficulty_step
#             self.difficulty_level -= 1
#             log_debug(f'decrease level, current {self.difficulty_level}')
#
#     def increase_level(self):
#         if self.difficulty < g_cfg.max_difficulty:
#             self.difficulty += g_cfg.difficulty_step
#             self.difficulty_level += 1
#             log_debug(f'increase level, current {self.difficulty_level}')
#
#     def register(self, episode_len, distance):  # FIXME: THIS DISTANCE IS ON CMD DIRECTION
#         self.counter += 1
#         if episode_len == g_cfg.max_sim_iterations:
#             self.miss = 0
#             self.combo += 1
#         else:
#             self.combo = 0
#             self.miss += 1
#         log_debug(f'Miss{self.miss} Combo{self.combo} distance{distance:.2f}')
#         if self.miss and self.miss % g_cfg.miss_threshold == 0:
#             self.decrease_level()
#             return True
#         elif self.combo and self.combo % g_cfg.combo_threshold == 0:
#             lower, upper = g_cfg.distance_threshold
#             if distance > upper:
#                 self.increase_level()
#                 return True
#             # elif distance < lower:
#             #     self.decreaseLevel()
#
#         return False
#
#     def reset(self):
#         self.terrain = makeStandardRoughTerrain(self.bullet_client, self.difficulty)
