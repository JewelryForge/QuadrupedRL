from collections import deque

import numpy as np

from burl.rl.reward import *
from burl.utils import g_cfg


# class RewardManager(object):
#     def __init__(self, *rewards_weights, storage=False):
#         self._rewards, self._weights = [], []
#         for r, w in rewards_weights:
#             self._rewards.append(r)
#             self._weights.append(w)
#         self._rewards, self._weights = np.array(self._rewards), np.array(self._weights)
#         self._weights = self._weights * 0.1 / self._weights.sum()
#         self.storage = storage
#         if storage:
#             self.reward_buffer = deque(maxlen=2000)
#             self.weighted_reward_buffer = deque(maxlen=2000)
#         self._details = {}
#
#     @property
#     def rewards(self):
#         return self._rewards
#
#     @property
#     def weights(self):
#         return self._weights
#
#     @property
#     def details(self):
#         return self._details
#
#     def calculate_weighted_rewards(self, *args):
#         assert len(args) == len(self._rewards)
#         rewards = [r(*arg) for r, arg in zip(self._rewards, args)]
#         self._details = dict(zip((r.__class__.__name__ for r in self._rewards), rewards))
#         weighted_rewards = [r * w for r, w in zip(rewards, self._weights)]
#         if self.storage:
#             self.reward_buffer.append(rewards)
#             self.weighted_reward_buffer.append(weighted_rewards)
#         return sum(weighted_rewards)
#
#     def analyse(self, painting=True):
#         pass


class BasicTask(object):
    rewards_weights = (
        (LinearVelocityReward(), 0.2),
        (AngularVelocityReward(), 0.05),
        (BodyHeightReward(), 0.03),
        (RedundantLinearPenalty(), 0.02),
        (RedundantAngularPenalty(), 0.03),
        (BodyPosturePenalty(), 0.03),
        (FootSlipPenalty(), 0.02),
        (SmallStridePenalty(), 0.02),
        (TargetMutationPenalty(), 0.02),
        (BodyCollisionPenalty(), 0.02),
        (TorquePenalty(), 0.01)
    )

    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        self._env = env
        self._cmd = cmd
        from burl.rl.curriculum import BasicTerrainManager
        self._terrain: BasicTerrainManager = None
        self._rewards, self._weights = [], []
        for r, w in self.rewards_weights:
            self._rewards.append(r)
            self._weights.append(w)
        self._weights = np.array(self._weights)
        self._weights = self._weights * 0.1 / self._weights.sum()
        self._details = {}

    @property
    def cmd(self):
        return self._cmd

    @property
    def robot(self):
        return self._env.robot

    def calculateReward(self):
        linear = self.robot.getBaseLinearVelocityInBaseFrame()
        angular = self.robot.getBaseAngularVelocityInBaseFrame()
        contact_states = self.robot.getContactStates()
        mutation = self._env.getActionMutation()
        x, y, z = self.robot.getBasePosition(False)
        body_height = z - self._env.getTerrainHeight(x, y)
        slip = sum(self.robot.getFootSlipVelocity())
        # print(self.robot.getFootSlipVelocity())
        strides = self.robot.getStrides()
        torques = self.robot.getLastAppliedTorques()
        orientation = self.robot.orientation
        args = (
            (self._cmd, linear),  # Linear Rew
            (self._cmd, angular),  # Angular Rew
            (body_height,),  # Height Rew
            (self._cmd, linear),  # Linear Pen
            (angular,),  # Angular Pen
            (orientation,),  # Posture Pen
            (slip,),  # Slip Pen
            (strides,),  # Small Stride Pen
            (mutation,),  # Target Mut Pen
            (contact_states,),  # Collision Pen
            (torques,)  # Torque Pen
        )

        assert len(args) == len(self._rewards)
        rewards = [r(*arg) for r, arg in zip(self._rewards, args)]
        self._details = dict(zip((r.__class__.__name__ for r in self._rewards), rewards))
        weighted_sum = sum([r * w for r, w in zip(rewards, self._weights)])
        # return sum(weighted_rewards)

        # reward = self._reward_manager.calculate_weighted_rewards(*args)
        # print(np.array(linear), np.array(angular))
        # print(Rpy.from_quaternion(orientation))
        # print(np.array(rewards))
        # print(np.array(weighted_rewards))
        # print()
        # logging.debug('VEL:', str(np.array(linear)), str(np.array(angular)))
        # logging.debug('REW:', str(np.array(rewards)))
        # logging.debug('WEIGHTED:', str(np.array(weighted_rewards)))
        return weighted_sum

    def getRewardDetails(self):
        return self._details

    def reset(self):
        pass

    def makeTerrain(self):
        from burl.rl.curriculum import PlainTerrainManager, TerrainCurriculum, FixedRoughTerrainManager
        if g_cfg.plain:
            self._terrain = PlainTerrainManager(self._env.client)
        elif g_cfg.use_trn_curriculum:
            self._terrain = TerrainCurriculum(self._env.client)
        else:
            self._terrain = FixedRoughTerrainManager(self._env.client, seed=2)
        return self._terrain

    def register(self, episode_len):
        distance = np.dot(self.robot.position, self._cmd)
        return self._terrain.register(episode_len, distance)

    def isFailed(self):  # TODO: CHANGE TIME_OUT TO NORMALLY FINISH
        rob, env = self.robot, self._env
        (x, y, z), (r, p, _) = rob.position, Rpy.from_quaternion(rob.orientation)
        est_terrain_height = np.mean([env.getTerrainHeight(x, y)
                                      for x, y in [*rob.getFootXYsInWorldFrame(), (x, y)]])
        z -= est_terrain_height
        h_lb, h_ub = g_cfg.safe_height_range
        if ((z < h_lb or z > h_ub) or
                (r < -np.pi / 3 or r > np.pi / 3) or
                rob.getBaseContactState()):
            return True
        joint_diff = rob.getJointPositions() - rob.STANCE_POSTURE
        if any(joint_diff > g_cfg.joint_angle_range) or any(joint_diff < -g_cfg.joint_angle_range):
            return True
        return False


class RandomCmdTask(BasicTask):
    def __init__(self, env, seed=None):
        np.random.seed(seed)
        angle = np.random.random() * 2 * np.pi
        super().__init__(env, (np.cos(angle), np.sin(angle), 0))

    def reset(self):
        angle = np.random.random() * 2 * np.pi
        self._cmd = (np.cos(angle), np.sin(angle), 0)
