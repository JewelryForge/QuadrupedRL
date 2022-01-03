import numpy as np
from burl.rl.reward import RewardRegistry
from burl.utils import g_cfg


class BasicTask(RewardRegistry):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        super().__init__(np.asarray(cmd), env, env.robot)
        # from burl.rl.curriculum import BasicTerrainManager
        # self._terrain: BasicTerrainManager = None
        for reward, weight in g_cfg.rewards_weights:
            self.register(reward, weight)

        self.setCoefficient(0.25)
        # self.setCoefficient(0.1 / self._weight_sum)

    @property
    def cmd(self):
        return self._cmd

    @property
    def env(self):
        return self._env

    @property
    def robot(self):
        return self._robot

    def calculateReward(self):
        if g_cfg.test_mode:
            self.sendOrPrint()
        return super().calculateReward()

    def sendOrPrint(self):
        from burl.sim import Quadruped, TGEnv
        from burl.rl.reward import (TrivialStridePenalty, BodyHeightReward, RollPitchRatePenalty,
                                    FootClearanceReward, HipAnglePenalty)

        from burl.utils import udp_pub
        cmd = self.cmd
        env: TGEnv = self.env
        rob: Quadruped = self.robot

        def wrap(reward_type):
            return reward_type().__call__(cmd, env, rob)

        # for s, b in zip(rob.getFootSlipVelocity(), self.buf):
        #     b.append(s)
        # print(np.array([np.mean(b) for b in self.buf]))
        # print(rob.getCostOfTransport())

        # r_rate, p_rate, _ = rob.getBaseRpyRate()
        # print(r_rate, p_rate, wrap(RollPitchRatePenalty)())
        # linear = rob.getBaseLinearVelocityInBaseFrame()
        # projected_velocity = np.dot(cmd[:2], linear[:2])
        # print(projected_velocity)
        # print(env.getSafetyHeightOfRobot(), wrap(BodyHeightReward)())
        # strides = [np.linalg.norm(s) for s in rob.getStrides()]
        # if any(s != 0.0 for s in strides):
        #     print(strides, wrap(SmallStridePenalty)())
        # if any(clearances := rob.getFootClearances()):
        #     print(clearances, wrap(FootClearanceReward))
        # print(wrap(HipAnglePenalty))
        # print(rob.getJointPositions()[(0,3,6,9),])
        # data = {'joint_pos': tuple(rob.getJointPositions()),
        #         'commands': tuple(env.getLastCommand().reshape(-1)),
        #         'joint_vel': tuple(rob.getJointVelocities()),
        #         'kp_part': tuple(rob._motor._kp_part),
        #         'kd_part': tuple(rob._motor._kd_part),
        #         'torque': tuple(rob.getLastAppliedTorques())
        #         }
        # udp_pub.send(data)

    def reset(self):
        pass

    # def curriculumUpdate(self, episode_len):
    #     distance = np.dot(self.robot.position, self._cmd)
    #     return self._terrain.register(episode_len, distance)

    def isFailed(self):  # TODO: CHANGE TIME_OUT TO NORMALLY FINISH
        r, _, _ = self._robot.rpy
        safety_h = self._env.getSafetyHeightOfRobot()
        h_lb, h_ub = self._robot.STANCE_HEIGHT * 0.5, self._robot.STANCE_HEIGHT * 1.5
        if (safety_h < h_lb or safety_h > h_ub or r < -np.pi / 3 or r > np.pi / 3 or
                self._robot.getBaseContactState()):
            return True
        joint_diff = self._robot.getJointPositions() - self._robot.STANCE_POSTURE
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
