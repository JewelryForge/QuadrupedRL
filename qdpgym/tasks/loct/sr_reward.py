from abc import ABC

import numpy as np

from qdpgym.utils import tf


class Reward(ABC):
    def __call__(self, robot, env, task) -> float:
        raise NotImplementedError


def exp_m2(v, k=1):
    return np.exp(-k * v ** 2)


class VelocityReward(Reward):
    def __init__(self, max_lin=1.0):
        self.max_lin = max_lin

    def __call__(self, robot, env, task):
        lin_cmd = task.cmd[:2]
        lin_vel = robot.get_velocimeter()[:2]
        lin_cmd_mag = tf.vnorm(lin_cmd) * self.max_lin
        if lin_cmd_mag != 0.:
            lin_cmd /= lin_cmd_mag  # unit

        prj_vel = np.dot(lin_cmd, lin_vel)
        ort_vel = tf.vnorm(lin_vel - lin_cmd * prj_vel)
        if lin_cmd_mag == 0.:
            lin_rew = exp_m2(tf.vnorm(lin_vel))
        elif prj_vel >= lin_cmd_mag:
            lin_rew = 1.
        else:
            lin_rew = exp_m2(lin_cmd_mag - prj_vel)

        ort_pen = exp_m2(ort_vel, 3)
        # print(lin_cmd_mag, prj_vel, lin_rew)
        return (lin_rew + ort_pen) / 2


class RotationReward(Reward):
    def __init__(self, max_ang=1.0):
        self.max_ang = max_ang

    def __call__(self, robot, env, task):
        ang_cmd = task.cmd[2]
        ang_vel = robot.get_base_rpy_rate()[2]
        ang_cmd_mag = abs(ang_cmd) * self.max_ang
        if ang_cmd == 0.:
            ang_rew = exp_m2(ang_vel)
        elif ang_vel * np.sign(ang_cmd) >= ang_cmd_mag:
            ang_rew = 1.
        else:
            ang_rew = exp_m2(ang_vel - ang_cmd)
        # print(ang_cmd_mag, ang_vel, ang_rew)
        return ang_rew


class BodyMotionPenalty(Reward):
    def __call__(self, robot, env, task):
        z_vel = robot.get_velocimeter()[2]
        r_rate, p_rate, _ = robot.get_base_rpy_rate()
        return -1.25 * z_vel ** 2 - 0.4 * abs(r_rate) - 0.4 * abs(p_rate)


class BodyCollisionPenalty(Reward):
    def __call__(self, robot, env, task):
        contact_sum = 0
        for i, contact in enumerate(robot.get_leg_contacts()):
            if contact and i % 3 != 2:
                contact_sum += 1
        return 1 - contact_sum


class JointMotionPenalty(Reward):
    def __call__(self, robot, env, task):
        vel_pen = -(robot.get_joint_vel() ** 2).sum()
        acc_pen = -(robot.get_joint_acc() ** 2).sum()
        return vel_pen + acc_pen * 1e-4


class TargetSmoothnessReward(Reward):
    def __call__(self, robot, env, task):
        action_history = env.action_history
        actions = [action_history[-i - 1] for i in range(3)]
        return -(
            ((actions[0] - actions[1]) ** 2).sum() +
            ((actions[0] - 2 * actions[1] + actions[2]) ** 2).sum()
        )


class TorquePenalty(Reward):
    def __call__(self, robot, env, task):
        return -(robot.get_last_torque() ** 2).sum()


class SlipPenalty(Reward):
    def __call__(self, robot, env, task):
        return -robot.get_slip_vel().sum()


class FootClearanceReward(Reward):
    def __call__(self, robot, env, task):
        return 1.
