from collections import deque
import numpy as np

from burl.rl.reward import *


class RewardManager(object):
    def __init__(self, *rewards_weights, storage=False):
        self._rewards, self._weights = [], []
        for r, w in rewards_weights:
            self._rewards.append(r)
            self._weights.append(w)
        self.storage = storage
        if storage:
            self.reward_buffer = deque(maxlen=2000)
            self.weighted_reward_buffer = deque(maxlen=2000)
        self._details = {}

    @property
    def rewards(self):
        return self._rewards

    @property
    def weights(self):
        return self._weights

    @property
    def details(self):
        return self._details

    def calculate_weighted_rewards(self, *args):
        assert len(args) == len(self._rewards)
        rewards = [r(*arg) for r, arg in zip(self._rewards, args)]
        self._details = dict(zip((r.__class__.__name__ for r in self._rewards), rewards))
        weighted_rewards = [r * w for r, w in zip(rewards, self._weights)]
        if self.storage:
            self.reward_buffer.append(rewards)
            self.weighted_reward_buffer.append(weighted_rewards)
        return sum(weighted_rewards)

    def analyse(self, painting=True):
        pass


class BasicTask(object):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        self._env = env
        self._robot = env.robot
        rewards = (
            (LinearVelocityReward(), 0.1),
            (AngularVelocityReward(), 0.05),
            (RedundantLinearPenalty(), 0.02),
            (RedundantAngularPenalty(), 0.03),
            (BodyPosturePenalty(), 0.03),
            (TargetMutationPenalty(), 0.03),
            (BodyHeightReward(), 0.02),
            (BodyCollisionPenalty(), 0.02),
            (TorquePenalty(), 0.03)
        )
        self._reward_names = [r.__class__.__name__ for r, _ in rewards]
        self._reward_manager = RewardManager(*rewards)
        self._cmd = cmd

    @property
    def cmd(self):
        return self._cmd

    @property
    def reward_names(self):
        return self._reward_names

    @property
    def num_rewards(self):
        return len(self._reward_names)

    def calculateReward(self):
        linear = self._robot.getBaseLinearVelocityInBaseFrame()
        angular = self._robot.getBaseAngularVelocityInBaseFrame()
        contact_states = self._robot.getContactStates()

        mutation = self._env.getActionMutation()
        x, y, z = self._robot.getBasePosition(False)
        body_height = z - self._env.getTerrainHeight(x, y)
        torques = self._robot.getLastAppliedTorques()
        orientation = self._robot.orientation
        args = (
            (self._cmd, linear),  # Linear Rew
            (self._cmd, angular),  # Angular Rew
            (self._cmd, linear),  # Linear Pen
            (angular,),  # Angular Pen
            (orientation,),  # Posture Pen
            (mutation,),  # Target Mut Pen
            (body_height,),  # Height Rew
            (contact_states,),  # Collision Pen
            (torques,)  # Torque Pen
        )
        reward = self._reward_manager.calculate_weighted_rewards(*args)
        # print(np.array(linear), np.array(angular))
        # print(Rpy.from_quaternion(orientation))
        # print(np.array(rewards))
        # print(np.array(weighted_rewards))
        # print()
        # logging.debug('VEL:', str(np.array(linear)), str(np.array(angular)))
        # logging.debug('REW:', str(np.array(rewards)))
        # logging.debug('WEIGHTED:', str(np.array(weighted_rewards)))
        return reward

    def getRewardDetails(self):
        return self._reward_manager.details

    def reset(self):
        pass

    MAX_MOVEMENT_ANGLE = 1.0

    def done(self):
        joint_diff = self._robot.getJointPositions(noisy=False) - self._robot.STANCE_POSTURE
        if any(joint_diff > self.MAX_MOVEMENT_ANGLE) or \
                any(joint_diff < -self.MAX_MOVEMENT_ANGLE):
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
