import math
from abc import ABC

import numpy as np

from qdpgym.utils import tf

ATANH0_95 = math.atanh(0.95)
ATANH0_9 = math.atanh(0.9)
SQRT_LOG10 = math.sqrt(math.log(10))
SQRT_LOG20 = math.sqrt(math.log(20))


class Reward(ABC):
    def __call__(self, robot, env, task) -> float:
        raise NotImplementedError


def tanh2_reshape(lower, upper):
    w = np.sqrt(ATANH0_95) / (upper - lower)

    def _reshape(v):
        return np.tanh(np.power((v - lower) * w, 2)) if v > lower else 0

    return _reshape


def tanh2_reverse(lower, upper, value):
    w = np.sqrt(ATANH0_95) / (upper - lower)
    return np.sqrt(np.arctanh(value)) / w + lower


def tanh_reshape(lower, upper):
    """
    Zoom tanh so that f(lower) = -0.9 and f(upper) = 0.9
    :return: function f
    """
    middle = (lower + upper) / 2
    w = ATANH0_9 / (upper - middle)

    def _reshape(v):
        return np.tanh((v - middle) * w)

    return _reshape


def tanh_reverse(lower, upper, value):
    middle = (lower + upper) / 2
    w = ATANH0_9 / (upper - middle)
    return np.arctanh(value) / w + middle


def exp_m2_reshape(upper):
    w = upper / SQRT_LOG10

    def _reshape(v):
        return np.exp(-(v / w) ** 2)

    return _reshape


def quadratic_linear_reshape(upper):
    k = 1 / upper ** 2
    kl = 2 * k * upper

    def _reshape(v):
        v = abs(v)
        return v ** 2 * k if v < upper else kl * (v - upper) + 1

    return _reshape


def soft_constrain(thr, upper):
    def _reshape(v):
        excess = abs(v) - thr
        if excess > 0:
            return -(excess / upper) ** 2
        return 0.

    return _reshape


class UnifiedLinearReward(Reward):
    def __init__(self, forward=1.5, lateral=1.0, ort_upper=0.3, ort_weight=0.33):
        self.forward, self.lateral = forward, lateral
        self.coeff = forward * lateral
        self.ort_reshape = tanh2_reshape(0, ort_upper)
        self.ort_weight = ort_weight

    def __call__(self, robot, env, task):
        lin_cmd = task.cmd[:2]
        lin_vel = robot.get_velocimeter()[:2]
        proj_vel = np.dot(lin_cmd, lin_vel)
        ort_vel = tf.vnorm(lin_vel - lin_cmd[:2] * proj_vel)
        # print(proj_vel, ort_vel)
        ort_pen = 1 - self.ort_reshape(ort_vel)
        if (lin_cmd == 0.).all():
            return ort_pen
        proj_rew = self.reshape(self.get_desired_velocity(lin_cmd), proj_vel)
        return (1 - self.ort_weight) * proj_rew + self.ort_weight * ort_pen

    def get_desired_velocity(self, cmd):
        return self.coeff / math.hypot(self.lateral * cmd[0], self.forward * cmd[1])

    @staticmethod
    def reshape(scale, value):
        return math.tanh(value * ATANH0_9 / scale)


class YawRateReward(Reward):
    def __init__(self, upper_pos=1.0, upper_neg=0.45):
        self.reshape_pos = tanh_reshape(-upper_pos, upper_pos)
        self.reshape_neg = quadratic_linear_reshape(upper_neg)

    def __call__(self, robot, env, task):
        yaw_cmd, yaw_rate = task.cmd[2], robot.get_base_rpy_rate()[2]
        if yaw_cmd != 0.0:
            return self.reshape_pos(yaw_rate / yaw_cmd)
        else:
            return 1 - self.reshape_neg(abs(yaw_rate))


class RollPitchRatePenalty(Reward):
    def __init__(self, dr_upper=1.0, dp_upper=0.5):
        self.dr_reshape = tanh2_reshape(0.0, dr_upper)
        self.dp_reshape = tanh2_reshape(0.0, dp_upper)

    def __call__(self, robot, env, task):
        r_rate, p_rate, _ = robot.get_base_rpy_rate()
        return 1 - (self.dr_reshape(abs(r_rate)) +
                    self.dp_reshape(abs(p_rate))) / 2


# class OrthogonalLinearPenalty(Reward):
#     def __init__(self, linear_upper=0.3):
#         self.reshape = tanh2_reshape(0, linear_upper)
#
#     def __call__(self, robot, env, task):
#         linear = robot.get_velocimeter()[:2]
#         v_o = np.asarray(linear) - np.asarray(cmd[:2]) * np.dot(linear, cmd[:2])
#         return 1 - self.reshape(tf.vnorm(v_o))


class VerticalLinearPenalty(Reward):
    def __init__(self, upper=0.4):
        self.reshape = quadratic_linear_reshape(upper)

    def __call__(self, robot, env, task):
        return 1 - self.reshape(robot.get_velocimeter()[2])


class BodyPosturePenalty(Reward):
    def __init__(self, roll_upper=np.pi / 12, pitch_upper=np.pi / 24):
        self.roll_reshape = quadratic_linear_reshape(roll_upper)
        self.pitch_reshape = quadratic_linear_reshape(pitch_upper)

    def __call__(self, robot, env, task):
        trnZ = env.get_interact_terrain_normal()
        robot_rot = robot.get_base_rot()
        trnY = tf.vcross(trnZ, robot_rot[:, 0])
        trnX = tf.vcross(trnY, trnZ)
        r, p, _ = tf.Rpy.from_rotation(np.array((trnX, trnY, trnZ)) @ robot_rot)
        return 1 - (self.roll_reshape(r) + self.pitch_reshape(p)) / 2


class BodyHeightReward(Reward):
    def __init__(self, des=0.4, range_=0.03):
        self.des = des
        self.reshape = quadratic_linear_reshape(range_)

    def __call__(self, robot, env, task):
        return env.get_relative_robot_height()


class ActionSmoothnessReward(Reward):
    def __init__(self, upper=400):
        self.reshape = exp_m2_reshape(upper)

    def __call__(self, robot, env, task):
        return self.reshape(env.get_action_accel()).sum() / 12


class JointMotionPenalty(Reward):
    def __init__(self, vel_upper=6, acc_upper=500):
        self.vel_reshape = np.vectorize(quadratic_linear_reshape(vel_upper))
        self.acc_reshape = np.vectorize(quadratic_linear_reshape(acc_upper))

    def __call__(self, robot, env, task):
        vel_pen = self.vel_reshape(robot.get_joint_vel()).sum()
        acc_pen = self.acc_reshape(robot.get_joint_acc()).sum()
        return 1 - (vel_pen + acc_pen) / 12


# class TorqueGradientPenalty(Reward):
#     def __init__(self, upper=200):
#         self.reshape = quadratic_linear_reshape(upper)
#
#     def __call__(self, cmd, env, robot):
#         torque_grad = robot.getTorqueGradients()
#         return -sum(self.reshape(grad) for grad in torque_grad) / 12


class FootSlipPenalty(Reward):
    def __init__(self, upper=0.5):
        self.reshape = np.vectorize(quadratic_linear_reshape(upper))

    def __call__(self, robot, env, task):
        return 1 - sum(self.reshape(robot.get_slip_vel()))


# class HipAnglePenalty(Reward):
#     def __init__(self, upper=0.3):
#         self.reshape = np.vectorize(quadratic_linear_reshape(upper))
#
#     def __call__(self, cmd, env, robot):
#         hip_angles = robot.getJointPositions()[(0, 3, 6, 9),]
#         return -sum(self.reshape(hip_angles)) / 4


class JointConstraintPenalty(Reward):
    def __init__(self, constraints=(0., 0., 0.), upper=(0.4, 0.6, 0.6)):
        self.hip_reshape = np.vectorize(soft_constrain(constraints[0], upper[0]))
        self.thigh_reshape = np.vectorize(soft_constrain(constraints[1], upper[1]))
        self.shank_reshape = np.vectorize(soft_constrain(constraints[2], upper[2]))

    def __call__(self, robot, env, task):
        joint_angles = robot.get_joint_pos() - robot.STANCE_CONFIG
        return (self.hip_reshape(joint_angles[((0, 3, 6, 9),)]).sum() +
                self.thigh_reshape(joint_angles[((1, 4, 7, 10),)]).sum() +
                self.shank_reshape(joint_angles[((2, 5, 8, 11),)]).sum()) / 12


# class TrivialStridePenalty(Reward):
#     def __init__(self, lower=-0.2, upper=0.4):
#         self.reshape = tanh_reshape(lower, upper)
#
#     def __call__(self, robot, env, task):
#         strides = [np.dot(s, cmd[:2]) for s in robot.get_strides()]
#         return sum(self.reshape(s) for s in strides if s != 0.0)
#         # return 1 - sum(1 - self.reshape(s) for s in strides if s != 0.0)


class FootClearanceReward(Reward):
    def __init__(self, upper=0.08):
        self.reshape = tanh2_reshape(0., upper)

    def __call__(self, robot, env, task):
        foot_clearances = robot.get_clearances()
        return sum(self.reshape(c) for c in foot_clearances)


class AliveReward(Reward):
    def __call__(self, robot, env, task):
        return 1.0


class ClearanceOverTerrainReward(Reward):
    def __call__(self, robot, env, task):
        reward = 0.
        for x, y, z in robot.get_foot_pos():
            terrain_extremum = env.arena.get_peak((x - 0.1, x + 0.1),
                                                  (y - 0.1, y + 0.1))[2]
            if z - robot.FOOT_RADIUS > terrain_extremum + 0.01:
                reward += 1 / 2
        return reward


class BodyCollisionPenalty(Reward):
    def __call__(self, robot, env, task):
        contact_sum = 0
        for i, contact in enumerate(robot.get_leg_contacts()):
            if contact and i % 3 != 2:
                contact_sum += 1
        return 1 - contact_sum


class TorquePenalty(Reward):
    def __init__(self, upper=900):
        self.reshape = np.vectorize(quadratic_linear_reshape(upper))

    def __call__(self, robot, env, task):
        return 1 - sum(self.reshape(robot.get_last_torque() ** 2))
