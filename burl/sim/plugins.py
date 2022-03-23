from __future__ import annotations

import numpy as np

from burl.utils import UdpPublisher

__all__ = ['Plugin', 'StatisticsCollector']


class Plugin(object):
    def on_init(self, task, robot, env):
        pass

    def on_simulation_step(self, task, robot, env):
        pass

    def on_step(self, task, robot, env) -> dict | None:
        pass

    def on_reset(self, task, robot, env):
        pass


class StatisticsCollector(Plugin):
    def __init__(self, publish=True):
        self._torque_sum = 0.
        self._torque_abs_sum = 0.
        self._torque_pen_sum = 0.0
        self._joint_motion_sum = 0.0
        self._sim_step_counter = 0
        self._step_counter = 0
        self._publish = publish
        if self._publish:
            self._udp_pub = UdpPublisher(9870)

    def on_simulation_step(self, task, robot, env):
        from burl.sim.env import Quadruped, FixedTgEnv
        from burl.rl.reward import TorquePenalty, JointMotionPenalty

        cmd = task.cmd
        env: FixedTgEnv
        rob: Quadruped = robot

        def wrap(reward_type):
            return reward_type()(cmd, env, rob)

        self._sim_step_counter += 1
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
        if self._publish:
            data = {
                'joint_states': {
                    'joint_pos': rob.getJointPositions().tolist(),
                    'commands': rob.getLastCommand().tolist(),
                    'joint_vel': rob.getJointVelocities().tolist(),
                    'joint_acc': rob.getJointAccelerations().tolist(),
                    # 'kp_part': tuple(rob._motor._kp_part),
                    # 'kd_part': tuple(rob._motor._kd_part),
                    'torque': rob.getLastAppliedTorques().tolist(),
                    'contact': rob.getContactStates().tolist()
                },
                'body_height': env.getTerrainBasedHeightOfRobot(),
                'cot': rob.getCostOfTransport(),
                'twist': {
                    'linear': rob.getBaseLinearVelocityInBaseFrame().tolist(),
                    'angular': rob.getBaseAngularVelocityInBaseFrame().tolist(),
                },
                'torque_pen': wrap(TorquePenalty)
            }
            self._udp_pub.send(data)

    def on_step(self, task, robot, env):
        self._step_counter += 1

    def on_reset(self, task, robot, env):
        print('episode len:', self._step_counter)
        print('cot', robot.getCostOfTransport())
        print('mse torque', np.sqrt(self._torque_sum / self._sim_step_counter))
        print('abs torque', self._torque_abs_sum / self._sim_step_counter)
        print('torque pen', self._torque_pen_sum / self._sim_step_counter)
        print('joint motion pen', self._joint_motion_sum / self._sim_step_counter)
        self._torque_sum = 0.
        self._torque_abs_sum = 0.
        self._torque_pen_sum = 0.0
        self._joint_motion_sum = 0.0
        self._sim_step_counter = 0
        self._step_counter = 0
