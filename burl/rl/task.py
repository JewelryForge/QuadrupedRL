import math
import random

import numpy as np

from burl.rl.curriculum import GameInspiredCurriculum, DisturbanceCurriculum
from burl.rl.reward import *
from burl.utils import g_cfg

__all__ = ['BasicTask', 'RandomForwardBackTask', 'RandomCmdTask', 'get_task']


class BasicTask(RewardRegistry):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        super().__init__(np.asarray(cmd), env, env.robot)
        for reward, weight in g_cfg.rewards_weights:
            self.register(reward, weight)

        self.setCoefficient(0.25)
        self.curriculums: list[GameInspiredCurriculum] = []
        if g_cfg.add_disturbance:
            self.addCurriculum(DisturbanceCurriculum(aggressive=True))
        # self.setCoefficient(0.1 / self._weight_sum)

    cmd = property(lambda self: self._cmd)
    env = property(lambda self: self._env)
    robot = property(lambda self: self._robot)

    def addCurriculum(self, curriculum: GameInspiredCurriculum):
        self.curriculums.append(curriculum)
        if g_cfg.test_mode:
            curriculum.maxLevel()

    def onInit(self):
        for cur in self.curriculums:
            cur.onReset(self._cmd, self._robot, self._env)

    def onSimulationStep(self):
        for cur in self.curriculums:
            cur.onSimulationStep(self._cmd, self._robot, self._env)
        if g_cfg.test_mode:
            self.collectStatistics()

    def onStep(self):
        info = {}
        for cur in self.curriculums:
            cur.onStep(self._cmd, self._robot, self._env)
            info[cur.__class__.__name__] = cur.difficulty_degree
        return info

    def collectStatistics(self):
        from burl.sim import Quadruped, FixedTgEnv

        from burl.utils import udp_pub
        from collections import deque
        cmd = self.cmd
        env: FixedTgEnv = self.env
        rob: Quadruped = self.robot

        def wrap(reward_type):
            return reward_type().__call__(cmd, env, rob)

        if not hasattr(self, '_torque_sum'):
            self._torque_sum = 0.
            self._torque_abs_sum = 0.
            self._torque_pen_sum = 0.0
        self._torque_sum += rob.getLastAppliedTorques() ** 2
        self._torque_abs_sum += abs(rob.getLastAppliedTorques())
        self._torque_pen_sum += wrap(TorquePenalty)

        # print(max(rob.getLastAppliedTorques()))
        # print(wrap(HipAnglePenalty))
        # print(rob.getBaseLinearVelocityInBaseFrame()[2])

        # print(wrap(TorquePenalty))
        # r_rate, p_rate, _ = rob.getBaseRpyRate()
        # print(r_rate, p_rate, wrap(RollPitchRatePenalty))
        # r, p, _ = rob.rpy
        # print(r, p, wrap(BodyPosturePenalty))
        # print(cmd, rob.getBaseLinearVelocityInBaseFrame()[:2], wrap(LinearVelocityReward))
        # print(env.getSafetyHeightOfRobot(), wrap(BodyHeightReward))
        # print(rob.getCostOfTransport(), wrap(CostOfTransportReward))
        # strides = [np.linalg.norm(s) for s in rob.getStrides()]
        # if any(s != 0.0 for s in strides):
        #     print(strides, wrap(SmallStridePenalty))
        # if any(clearances := rob.getFootClearances()):
        #     print(clearances, wrap(FootClearanceReward))
        data = {
            'joint_states': {
                'joint_pos': tuple(rob.getJointPositions()),
                'commands': tuple(rob._command_history[-1]),
                'joint_vel': tuple(rob.getJointVelocities()),
                # 'kp_part': tuple(rob._motor._kp_part),
                # 'kd_part': tuple(rob._motor._kd_part),
                'torque': tuple(rob.getLastAppliedTorques()),
                'contact': tuple(rob.getContactStates())
            },
            'body_height': env.getSafetyHeightOfRobot(),
            'cot': rob.getCostOfTransport(),
            'twist': {
                'linear': tuple(rob.getBaseLinearVelocityInBaseFrame()),
                'angular': tuple(rob.getBaseAngularVelocityInBaseFrame()),
            },
            'torque_pen': wrap(TorquePenalty)
        }
        udp_pub.send(data)

    def reset(self):
        for cur in self.curriculums:
            cur.onReset(self._cmd, self._robot, self._env)
        if g_cfg.test_mode:
            print('cot', self.robot.getCostOfTransport())
            print('mse torque', np.sqrt(self._torque_sum / self.robot._step_counter))
            print('abs torque', self._torque_abs_sum / self.robot._step_counter)
            print('torque pen', self._torque_pen_sum / self.robot._step_counter)
            self._torque_sum = 0.
            self._torque_abs_sum = 0.
            self._torque_pen_sum = 0.0

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


class RandomForwardBackTask(BasicTask):
    def __init__(self, env, seed=None):
        random.seed(seed)
        self.update_interval = 500
        super().__init__(env, self.random_cmd())

    @staticmethod
    def random_cmd():
        return np.array((random.choice((1., -1.)), 0., 0.))

    def reset(self):
        self._cmd = self.random_cmd()

    def onStep(self):
        if self._env.sim_step % self.update_interval == 0:
            self.reset()
        super().onStep()


class RandomLinearCmdTask(RandomForwardBackTask):
    @staticmethod
    def random_cmd():
        yaw = random.uniform(0, 2 * np.pi)
        return np.array((math.cos(yaw), math.sin(yaw), 0))


class RandomCmdTask(RandomForwardBackTask):
    @staticmethod
    def random_cmd():
        yaw = random.uniform(0, 2 * np.pi)
        return np.array((math.cos(yaw), math.sin(yaw), random.choice((-1., 0, 1.))))


def get_task(task_type: str):
    if task_type == 'basic':
        return BasicTask
    elif task_type == 'randFB':
        return RandomForwardBackTask
    elif task_type == 'randLn':
        return RandomLinearCmdTask
    elif task_type == 'randCmd':
        return RandomCmdTask
    else:
        raise RuntimeError(f"Unknown task type '{task_type}'")
