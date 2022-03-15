import math
import random

import numpy as np

from burl.rl.curriculum import CURRICULUM_PROTOTYPE, CURRICULUM_DISTRIB
from burl.rl.reward import *
from burl.utils import g_cfg

__all__ = ['BasicTask', 'RandomLinearCmdTask', 'RandomCmdTask', 'get_task', 'CentralizedTask']


class BasicTask(RewardRegistry):
    def __init__(self, env, cmd=(1.0, 0.0, 0.0)):
        super().__init__(np.asarray(cmd), env, env.robot)
        for reward, weight in g_cfg.rewards_weights:
            self.addReward(reward, weight)
        self.setCoefficient(0.25)
        # self.setCoefficient(0.1 / self._weight_sum)

        self.curricula: list[CURRICULUM_DISTRIB] = []

    cmd = property(lambda self: self._cmd)
    env = property(lambda self: self._env)
    robot = property(lambda self: self._robot)

    def makeTerrain(self, terrain_type):
        from burl.sim import Plain
        if terrain_type == 'plain':
            terrain_inst = Plain()
            terrain_inst.spawn(self._env.client)
        elif terrain_type == 'curriculum':
            for crm in self.curricula:
                if hasattr(crm, 'generateTerrain'):
                    terrain_inst = crm.generateTerrain(self._env.client)
                    break
            else:
                raise RuntimeError('Not a TerrainCurriculum instance')
        elif terrain_type == 'rough':
            raise NotImplementedError
        elif terrain_type == 'slope':
            raise NotImplementedError
        else:
            raise RuntimeError(f'Unknown terrain type {terrain_type}')
        return terrain_inst

    def addCurriculum(self, curriculum: CURRICULUM_DISTRIB):
        self.curricula.append(curriculum)

    def onInit(self):
        for crm in self.curricula:
            crm.onInit(self, self._robot, self._env)

    def onSimulationStep(self):
        for crm in self.curricula:
            crm.onSimulationStep(self, self._robot, self._env)
        if g_cfg.test_mode:
            self.collectStatistics()

    def onStep(self):
        info = {}
        for crm in self.curricula:
            crm.onStep(self, self._robot, self._env)
            info[crm.__class__.__name__] = crm.difficulty_degree
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
            self._joint_motion_sum = 0.0
        self._torque_sum += rob.getLastAppliedTorques() ** 2
        self._torque_abs_sum += abs(rob.getLastAppliedTorques())
        self._torque_pen_sum += wrap(TorquePenalty)
        self._joint_motion_sum += wrap(JointMotionPenalty)
        # print(wrap(LinearVelocityReward))
        # print(rob.getJointVelocities())
        # print(rob.getJointAccelerations())
        # print()
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
                'joint_acc': tuple(rob.getJointAccelerations()),
                # 'kp_part': tuple(rob._motor._kp_part),
                # 'kd_part': tuple(rob._motor._kd_part),
                'torque': tuple(rob.getLastAppliedTorques()),
                'contact': tuple(rob.getContactStates())
            },
            'body_height': env.getTerrainBasedHeightOfRobot(),
            'cot': rob.getCostOfTransport(),
            'twist': {
                'linear': tuple(rob.getBaseLinearVelocityInBaseFrame()),
                'angular': tuple(rob.getBaseAngularVelocityInBaseFrame()),
            },
            'torque_pen': wrap(TorquePenalty)
        }
        udp_pub.send(data)

    def reset(self):
        for crm in self.curricula:
            crm.onReset(self, self._robot, self._env)
        if g_cfg.test_mode:
            print('cot', self.robot.getCostOfTransport())
            print('mse torque', np.sqrt(self._torque_sum / self.robot._step_counter))
            print('abs torque', self._torque_abs_sum / self.robot._step_counter)
            print('torque pen', self._torque_pen_sum / self.robot._step_counter)
            print('joint motion pen', self._joint_motion_sum / self.robot._step_counter)
            self._torque_sum = 0.
            self._torque_abs_sum = 0.
            self._torque_pen_sum = 0.0
            self._joint_motion_sum = 0.0

    def isFailed(self):  # TODO: CHANGE TIME_OUT TO NORMALLY FINISH
        r, _, _ = self._robot.rpy
        safety_h = self._env.getTerrainBasedHeightOfRobot()
        h_lb, h_ub = self._robot.STANCE_HEIGHT * 0.5, self._robot.STANCE_HEIGHT * 1.5
        if (safety_h < h_lb or safety_h > h_ub or r < -np.pi / 3 or r > np.pi / 3 or
                self._robot.getBaseContactState()):
            return True
        # joint_diff = self._robot.getJointPositions() - self._robot.STANCE_POSTURE
        # if any(joint_diff > g_cfg.joint_angle_range) or any(joint_diff < -g_cfg.joint_angle_range):
        #     return True
        return False


# class RandomLeftRightTask(BasicTask):
#     def __init__(self, env):
#         self.update_interval = 1500
#         self.last_update = 0
#         self.last_cmd = 0
#         super().__init__(env, (0., 1., 0.))
#
#     def reset(self):
#         self.last_update = 0
#         self._cmd = np.array((0., 1., 0.))
#         super().reset()
#
#     def onStep(self):
#         if self._env.sim_step >= self.last_update + self.update_interval:
#             self._cmd = np.array((0., 1., 0.) if self.last_cmd else (0., -1., 0.))
#             self.last_cmd = 1 - self.last_cmd
#             self.last_update = self._env.sim_step
#         super().onStep()


class RandomLinearCmdTask(BasicTask):
    def __init__(self, env, seed=None):
        random.seed(seed)
        # self.stop_prob = 0.2
        self.interval_range = (1000, 2500)
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        super().__init__(env, self.random_cmd())

    def random_cmd(self):
        # if random.random() < self.stop_prob:
        #     return np.array((0., 0., 0.))
        yaw = random.uniform(0, 2 * np.pi)
        return np.array((math.cos(yaw), math.sin(yaw), 0))

    def reset(self):
        self.update_interval = random.uniform(*self.interval_range)
        self.last_update = 0
        self._cmd = self.random_cmd()
        super().reset()

    def onStep(self):
        if self._env.sim_step >= self.last_update + self.update_interval:
            self._cmd = self.random_cmd()
            self.last_update = self._env.sim_step
            self.update_interval = random.uniform(*self.interval_range)
        return super().onStep()


class RandomCmdTask(RandomLinearCmdTask):
    def random_cmd(self):
        yaw = random.uniform(0, 2 * np.pi)
        return np.array((math.cos(yaw), math.sin(yaw), random.choice((-1., 0, 1.))))
        # return np.array((math.cos(yaw), math.sin(yaw), clip(random.gauss(0, 0.5), -1, 1)))


class CentralizedTask(object):
    def __init__(self):
        self.curriculum_prototypes: list[CURRICULUM_PROTOTYPE] = []
        aggressive = g_cfg.test_mode
        buffer_len = g_cfg.num_envs * 2
        if g_cfg.use_centralized_curriculum:
            from burl.rl.curriculum import CentralizedDisturbanceCurriculum, CentralizedTerrainCurriculum
            if g_cfg.add_disturbance:
                self.curriculum_prototypes.append(
                    CentralizedDisturbanceCurriculum(buffer_len=buffer_len, aggressive=aggressive))
            if g_cfg.trn_type == 'curriculum':
                self.curriculum_prototypes.append(
                    CentralizedTerrainCurriculum(buffer_len=buffer_len, aggressive=aggressive))
        else:
            from burl.rl.curriculum import DisturbanceCurriculum, TerrainCurriculum
            if g_cfg.add_disturbance:
                self.curriculum_prototypes.append(DisturbanceCurriculum(aggressive))
            if g_cfg.trn_type == 'curriculum':
                self.curriculum_prototypes.append(TerrainCurriculum(aggressive))

    def makeDistribution(self, task_class, *args, **kwargs):
        def _makeDistribution(env):
            task_inst: BasicTask = task_class(env, *args, **kwargs)
            for crm in self.curriculum_prototypes:
                task_inst.addCurriculum(crm.makeDistribution())
            return task_inst

        return _makeDistribution

    def updateCurricula(self):
        for crm in self.curriculum_prototypes:
            crm.checkLetter()


def get_task(task_type: str):
    if task_type == 'basic':
        return BasicTask
    elif task_type == 'randLn':
        return RandomLinearCmdTask
    elif task_type == 'randCmd':
        return RandomCmdTask
    # elif task_type == 'randLR':
    #     return RandomLeftRightTask
    else:
        raise RuntimeError(f"Unknown task type '{task_type}'")
