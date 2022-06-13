import collections
import math
import os
import os.path
from dataclasses import field, dataclass
from typing import Optional, Deque, Union, List, Tuple, Any

import numpy as np
import pybullet as pyb

from qdpgym import tf, utils as ut
from qdpgym.sim import rsc_dir
from qdpgym.sim.abc import ARRAY_LIKE, Terrain, Quadruped, QuadrupedHandle
from qdpgym.sim.abc import Snapshot, Command
from qdpgym.sim.blt.utils import DynamicsInfo
from qdpgym.sim.common.motor import PdMotorSim, ActuatorNetSim
from qdpgym.sim.common.noisyhandle import NoisyHandle


class AliengoModelBt(object):
    MODEL_PATH = 'aliengo/model/aliengo.urdf'

    LEG_NAMES = ('FR', 'FL', 'RR', 'RL')
    JOINT_TYPES = ('hip', 'thigh', 'calf', 'foot')
    JOINT_SUFFIX = ('joint', 'joint', 'joint', 'fixed')

    def __init__(self):
        self._body_id: int = -1
        self._num_joints: int = 0
        self._joint_names: Optional[List[str, ...]] = None
        self._joint_ids: Optional[List[int]] = None
        self._motor_ids: Optional[List[int]] = None
        self._foot_ids: Optional[List[int]] = None

        self._base_dyn: Optional[DynamicsInfo] = None
        self._leg_dyns: Optional[List[DynamicsInfo]] = None
        self._mass: Optional[float] = None

    @property
    def motor_ids(self):
        return self._motor_ids

    @property
    def mass(self):
        return self._mass

    @property
    def foot_ids(self):
        return self._foot_ids

    def spawn(self, sim_env, init_pose):
        model_path = os.path.join(rsc_dir, self.MODEL_PATH)
        self._body_id = sim_env.loadURDF(model_path, *init_pose)  # , flags=flags)
        self._num_joints = sim_env.getNumJoints(self._body_id)
        self._joint_names = ['_'.join((leg, j, s)) for leg in self.LEG_NAMES
                             for j, s in zip(self.JOINT_TYPES, self.JOINT_SUFFIX)]

        joint_name_to_id = {}
        for i in range(sim_env.getNumJoints(self._body_id)):
            joint_info = sim_env.getJointInfo(self._body_id, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        self._joint_ids = [joint_name_to_id.get(n, -1) for n in self._joint_names]
        self._motor_ids = [self._get_joint_id(leg, j) for leg in range(4) for j in range(3)]
        self._foot_ids = [self._get_joint_id(leg, -1) for leg in range(4)]
        sim_env.setPhysicsEngineParameter(enableConeFriction=0)
        for foot_id in self._foot_ids:
            sim_env.enableJointForceTorqueSensor(self._body_id, foot_id, True)

        self._base_dyn = DynamicsInfo(sim_env.getDynamicsInfo(self._body_id, 0))
        self._leg_dyns = [DynamicsInfo(sim_env.getDynamicsInfo(self._body_id, i))
                          for i in self._motor_ids[:3]]
        return self._body_id, self._motor_ids, self._foot_ids

    def init_episode_dynamics(self, sim_env, random_state=None):
        def _change_dyn(_link_id, *args, **kwargs):
            sim_env.changeDynamics(self._body_id, _link_id, *args, **kwargs)

        if random_state is None:
            for link_id in self._foot_ids:
                _change_dyn(link_id, lateralFriction=1.0, spinningFriction=0.2)
        else:
            base_mass = self._base_dyn.mass * random_state.uniform(0.8, 1.2)
            base_inertia = self._base_dyn.inertia * random_state.uniform(0.8, 1.2, 3)
            _change_dyn(0, mass=base_mass, localInertiaDiagonal=base_inertia)

            leg_masses, leg_inertia = [], []
            for _ in range(4):
                for leg_dyn in self._leg_dyns:
                    leg_masses.append(leg_dyn.mass * random_state.uniform(0.8, 1.2))
                    leg_inertia.append(leg_dyn.inertia * random_state.uniform(0.8, 1.2, 3))
            for link_id, mass, inertia in zip(self._motor_ids, leg_masses, leg_inertia):
                _change_dyn(link_id, mass=mass, localInertiaDiagonal=inertia)

            for link_id, fric in zip(self._foot_ids, random_state.uniform(0.4, 1.0, 4)):
                _change_dyn(link_id, lateralFriction=fric, spinningFriction=0.2)

        for link_id in self._motor_ids:
            _change_dyn(link_id, linearDamping=0, angularDamping=0)
        sim_env.setJointMotorControlArray(self._body_id, self._motor_ids,
                                          pyb.VELOCITY_CONTROL, forces=(0.025,) * 12)

        self._mass = sum([sim_env.getDynamicsInfo(self._body_id, i)[0]
                          for i in range(self._num_joints)])

    def _get_joint_id(self, leg: Union[int, str], joint_type: Union[int, str] = 0):
        if joint_type < 0:
            joint_type += 4
        return self._joint_ids[leg * 4 + joint_type]


class Aliengo(Quadruped):
    LINK_LENGTHS = (0.083, 0.25, 0.25)
    HIP_OFFSETS = ((0.2399, -0.051, 0.), (0.2399, 0.051, 0.),
                   (-0.2399, -0.051, 0), (-0.2399, 0.051, 0.))
    FOOT_RADIUS = 0.0265
    ROBOT_SIZE = ((-0.325, 0.325), (-0.155, 0.155))

    STANCE_HEIGHT = 0.43
    STANCE_CONFIG = (0., 0.6435, -1.287) * 4
    STANCE_FOOT_POSITIONS = ((0., -LINK_LENGTHS[0], -0.4), (0., LINK_LENGTHS[0], -0.4),
                             (0., -LINK_LENGTHS[0], -0.4), (0., LINK_LENGTHS[0], -0.4))
    JOINT_LIMITS = ((-1.22, 1.22), (None, None), (-2.77, -0.7)) * 4
    TORQUE_LIMITS = 44.4

    @dataclass
    class LocomotionInfo:
        time: float = 0.
        last_stance_states: Any = field(default_factory=lambda: [None] * 4)
        max_foot_heights: np.ndarray = field(default_factory=lambda: np.zeros(4))
        foot_clearances: np.ndarray = field(default_factory=lambda: np.zeros(4))
        strides: List[Tuple[float, float]] = field(default_factory=lambda: [(0., 0.)] * 4)
        slips: List[float] = field(default_factory=lambda: [0.] * 4)

    def __init__(self, frequency: int, motor: str, noisy=True):
        self._freq = frequency
        if motor == 'pd':
            self._motor = PdMotorSim(self._freq, 150, 4)
        elif motor == 'actuator_net':
            self._motor = ActuatorNetSim(self._freq)
            self._motor.load_params(os.path.join(rsc_dir, 'acnet_220526.pt'))
        else:
            raise NotImplementedError
        self._motor.set_joint_limits(*zip(*self.JOINT_LIMITS))
        self._motor.set_torque_limits(self.TORQUE_LIMITS)
        self._noisy_on = noisy

        self._sim_env = None if False else pyb
        self._body_id = -1
        self._motor_ids: Optional[List[int]] = None
        self._foot_ids: Optional[List[int]] = None
        self._model = AliengoModelBt()

        self._noisy: Optional[NoisyHandle] = None
        if self._noisy_on:
            self._noisy = NoisyHandle(self, frequency)

        self._state: Optional[Snapshot] = None
        self._state_history: Deque[Snapshot] = collections.deque(maxlen=100)
        self._cmd: Optional[Command] = None
        self._cmd_history: Deque[Command] = collections.deque(maxlen=100)
        self._locom: Optional[Aliengo.LocomotionInfo] = None

        self._init_yaw = 0.
        self._init_pose = np.array((0., 0., self.STANCE_HEIGHT)), np.array((0., 0., 0., 1.))
        self._random_dynamics = False
        self._latency_range = None

    @property
    def noisy(self) -> QuadrupedHandle:
        return self._noisy if self._noisy_on else self

    @property
    def id(self):
        return self._body_id

    @property
    def obs_history(self):
        return ut.PadWrapper(self._state_history)

    @property
    def cmd_history(self):
        return ut.PadWrapper(self._cmd_history)

    def set_init_pose(self, x=0., y=0., yaw=0.):
        """
        Set init pose before adding to an arena.
        The pose may be modified by `add_to` to fit a terrain.
        """

        self._init_pose[0][0] = x
        self._init_pose[0][1] = y
        self._init_yaw = yaw

    def add_to(self, arena: Terrain):
        foot_xy = (np.array(self.STANCE_FOOT_POSITIONS) + np.array(self.HIP_OFFSETS))[:, :2]
        if self._init_yaw != 0.:
            cy, sy = np.cos(self._init_yaw), np.sin(self._init_yaw)
            trans = np.array(((cy, -sy),
                              (sy, cy)))
            foot_xy = np.array([trans @ f_xy for f_xy in foot_xy])
        else:
            cy, sy = 1., 0.
        x, y, _ = self._init_pose[0]
        foot_xy += (x, y)

        terrain_points = []
        est_height = 0.
        for x, y in foot_xy:
            z = arena.get_height(x, y)
            est_height += z
            terrain_points.append((x, y, z))
        est_height /= 4
        init_height = est_height + self.STANCE_HEIGHT

        trn_Z = tf.estimate_normal(terrain_points)
        trn_Y = tf.vcross(trn_Z, (cy, sy, 0.))
        trn_X = tf.vcross(trn_Y, trn_Z)
        orn = tf.Quaternion.inverse(
            tf.Quaternion.from_rotation(np.array((trn_X, trn_Y, trn_Z)))
        )
        self._init_pose = np.array((x, y, init_height)), orn

    def spawn(self, sim_env, random_state, cfg=None):
        self._sim_env = sim_env

        # flags = pyb.URDF_USE_SELF_COLLISION if self._self_collision else 0
        if self._body_id == -1:
            self._body_id, self._motor_ids, self._foot_ids = \
                self._model.spawn(self._sim_env, self._init_pose)
        else:
            sim_env.resetBasePositionAndOrientation(self._body_id, *self._init_pose)
            sim_env.resetBaseVelocity(self._body_id, (0.,) * 3, (0.,) * 3)
        if self._random_dynamics:
            self._model.init_episode_dynamics(sim_env, random_state)
        else:
            self._model.init_episode_dynamics(sim_env)

        if cfg is None:
            cfg = self.STANCE_CONFIG
        self._configure_joints(cfg)
        if self._noisy_on and self._latency_range is not None:
            self._noisy.latency = random_state.uniform(*self._latency_range)

        self._state = self._cmd = None
        self._state_history.clear()
        self._cmd_history.clear()
        self._locom = Aliengo.LocomotionInfo()
        if self._noisy is not None:
            self._noisy.reset()

    def spawn_on_rack(self, sim_env, random_state):
        self._sim_env = sim_env

        # flags = pyb.URDF_USE_SELF_COLLISION if self._self_collision else 0
        assert self._body_id == -1
        self._body_id, self._motor_ids, self._foot_ids = \
            self._model.spawn(
                self._sim_env,
                ((0., 0., 1.), (0., 0., 0., 1.))
            )
        self._model.init_episode_dynamics(sim_env)
        self._configure_joints(self.STANCE_CONFIG)

        sim_env.createConstraint(
            self._body_id, -1, -1, -1,
            jointType=pyb.JOINT_FIXED,
            jointAxis=(0., 0., 0.),
            parentFramePosition=(0., 0., 0.),
            childFramePosition=(0., 0., 1.)
        )

        if self._noisy_on and self._latency_range is not None:
            self._noisy.latency = random_state.uniform(*self._latency_range)

        self._state = self._cmd = None
        self._state_history.clear()
        self._cmd_history.clear()
        self._locom = Aliengo.LocomotionInfo()
        if self._noisy is not None:
            self._noisy.reset()

    def update_observation(self, random_state, minimal=False):
        """Get robot states from pybullet"""
        if minimal:
            # Only update motor observations, for basic locomotion
            joint_pos, joint_vel = np.array(list(zip(
                *self._sim_env.getJointStates(self._body_id, self._motor_ids)
            ))[:2])
            self._motor.update_observation(joint_pos, joint_vel)
            return

        s = self._state = Snapshot()
        s.position, s.orientation = self._sim_env.getBasePositionAndOrientation(self._body_id)
        s.rotation = tf.Rotation.from_quaternion(s.orientation)
        s.rpy = tf.Rpy.from_quaternion(s.orientation)
        s.linear_vel, s.angular_vel = self._sim_env.getBaseVelocity(self._body_id)
        s.joint_pos, s.joint_vel = np.array(list(zip(
            *self._sim_env.getJointStates(self._body_id, self._motor_ids)
        ))[:2])

        rotation = s.rotation.T
        s.velocimeter = rotation @ s.linear_vel
        s.gyro = rotation @ s.angular_vel
        # accelerometer
        s.foot_pos, _, s.contact_forces = self._get_foot_states()
        s.torso_contact, s.leg_contacts = self._get_contact_states()

        s.rpy_rate = tf.get_rpy_rate_from_ang_vel(s.rpy, s.angular_vel)
        s.force_sensor = np.array(
            [rotation @ force for force in s.contact_forces]
        )

        if self._noisy_on:
            self._noisy.update_observation(self._state, random_state)
        self._state_history.append(self._state)

        # self._motor.update_observation(self._state.joints_pos, self._state.joints_vel)
        if self._noisy_on:
            n = self._noisy.not_delayed
            self._motor.update_observation(n.joint_pos, n.joint_vel)
        else:
            self._motor.update_observation(s.joint_pos, s.joint_vel)

        l = self._locom
        l.time += 1 / self._freq
        rolling_vel = s.joint_vel[((1, 4, 7, 10),)] + s.joint_vel[((2, 5, 8, 11),)]
        for i, (contact, foot_pos, rv) in enumerate(
            zip(s.leg_contacts[((2, 5, 8, 11),)], s.foot_pos, rolling_vel)
        ):
            if not contact:
                l.strides[i] = (0., 0.)
                l.slips[i] = l.foot_clearances[i] = 0.
                l.max_foot_heights[i] = max(l.max_foot_heights[i], foot_pos[2])
                continue
            if l.last_stance_states[i] is not None:
                time, pos = l.last_stance_states[i]
                duration = l.time - time
                if duration >= 0.05:
                    # Take as stride
                    l.strides[i] = (foot_pos - pos)[:2]
                    l.slips[i] = 0.
                    l.foot_clearances[i] = l.max_foot_heights[i] - pos[2]
                else:
                    # Take as slip and estimate slip velocity
                    l.slips[i] = abs(tf.vnorm((foot_pos - pos)[:2]) -
                                     self.FOOT_RADIUS * rv * duration)
                    l.strides[i] = (0., 0.)
                    l.foot_clearances[i] = 0.
            l.max_foot_heights[i] = foot_pos[2]
            l.last_stance_states[i] = (l.time, foot_pos)

    def apply_command(self, motor_commands: ARRAY_LIKE):
        """
        Calculate desired joint torques of position commands with motor model.
        :param motor_commands: array of desired motor positions.
        :return: array of desired motor torques.
        """
        motor_commands = np.asarray(motor_commands)
        torques = self._motor.apply_position(motor_commands)
        self._sim_env.setJointMotorControlArray(self._body_id, self._motor_ids,
                                                pyb.TORQUE_CONTROL, forces=torques)
        self._cmd = Command(
            command=motor_commands,
            torque=torques
        )
        self._cmd_history.append(self._cmd)
        return torques

    def apply_torques(self, torques):
        self._sim_env.setJointMotorControlArray(self._body_id, self._motor_ids,
                                                pyb.TORQUE_CONTROL, forces=torques)
        self._cmd = Command(torque=torques)
        self._cmd_history.append(self._cmd)

    def set_random_dynamics(self, flag: bool = True):
        self._random_dynamics = flag

    def set_latency(self, lower: float = None, upper: float = None):
        if lower is None:
            self._latency_range = None
        elif upper is None:
            self._noisy.latency = lower
        else:
            self._latency_range = (lower, upper)

    def get_base_pos(self) -> np.ndarray:
        return self._state.position

    def get_base_orn(self) -> np.ndarray:
        return self._state.orientation

    def get_base_rot(self) -> np.ndarray:
        return self._state.rotation

    def get_base_rpy(self) -> np.ndarray:
        return self._state.rpy

    def get_base_rpy_rate(self) -> np.ndarray:
        return self._state.rpy_rate

    def get_base_lin(self) -> np.ndarray:
        return self._state.linear_vel

    def get_base_ang(self) -> np.ndarray:
        return self._state.angular_vel

    def get_velocimeter(self):
        return self._state.velocimeter

    def get_gyro(self):
        return self._state.gyro

    def get_state_history(self, latency: float):
        return ut.get_padded(self._state_history, -int(latency * self._freq) - 1)

    def get_cmd_history(self, latency: float):
        return ut.get_padded(self._cmd_history, -int(latency * self._freq) - 1)

    def get_torso_contact(self):
        return self._state.torso_contact

    def get_leg_contacts(self):
        return self._state.leg_contacts

    def get_foot_pos(self):
        return self._state.foot_pos

    def get_foot_contacts(self):
        return self._state.leg_contacts[((2, 5, 8, 11),)]

    def get_contact_forces(self):
        return self._state.contact_forces

    def get_force_sensor(self):
        return self._state.force_sensor

    def get_slip_vel(self):
        return np.array(self._locom.slips) * self._freq

    def get_strides(self):
        return np.array(self._locom.strides)

    def get_clearances(self):
        return self._locom.foot_clearances

    def get_joint_pos(self) -> np.ndarray:
        return self._state.joint_pos

    def get_joint_vel(self) -> np.ndarray:
        return self._state.joint_vel

    def get_joint_acc(self) -> np.ndarray:
        if len(self._state_history) > 2:
            prev_joint_vel = self._state_history[-2].joint_vel
            return (self._state.joint_vel - prev_joint_vel) * self._freq

        return np.zeros(12)

    def get_last_command(self) -> np.ndarray:
        return self._cmd.command if self._cmd is not None else None

    def get_last_torque(self) -> np.ndarray:
        return self._cmd.torque if self._cmd is not None else None

    def _configure_joints(self, cfg):
        for i in range(12):
            self._sim_env.resetJointState(self._body_id, self._motor_ids[i], cfg[i], 0.0)

    def _get_foot_states(self):
        """Get foot positions, orientations and forces by getLinkStates and getContactPoints."""
        foot_states = self._sim_env.getLinkStates(self._body_id, self._foot_ids)
        foot_pos, foot_orn = [], []
        for foot_state in foot_states:
            foot_pos.append(foot_state[0])
            foot_orn.append(foot_state[1])

        foot_forces = []
        for foot_id in self._foot_ids:
            contact_points = self._sim_env.getContactPoints(bodyA=self._body_id, linkIndexA=foot_id)
            foot_force = np.zeros(3)
            for p in contact_points:
                for axis_idx, force_idx in zip((7, 11, 13), (9, 10, 12)):
                    axis = p[axis_idx]
                    force = p[force_idx]
                    foot_force += np.array(axis) * force
            foot_forces.append(foot_force)
        return np.array(foot_pos), np.array(foot_orn), np.array(foot_forces)

    def _get_contact_states(self):
        def _get_contact_state(link_id):
            return bool(self._sim_env.getContactPoints(bodyA=self._body_id, linkIndexA=link_id))

        torso_contact = _get_contact_state(0)
        leg_contact = []
        for leg in range(4):
            torso_contact = torso_contact or _get_contact_state(leg * 4 + 2)
            leg_contact.extend([_get_contact_state(leg * 4 + i) for i in range(3, 6)])
        return torso_contact, np.array(leg_contact)

    @classmethod
    def inverse_kinematics(cls, leg: int, pos: ARRAY_LIKE):
        """
        Calculate analytical inverse kinematics of certain leg, unconsidered about joint angle limits.
        Currently, only positions beneath the robot are supported.
        leg: 1, 3 for left and 0, 2 for right
        """
        shoulder_len, thigh_len, shank_len = cls.LINK_LENGTHS
        if leg % 2 == 1:
            shoulder_len *= -1
        pos = np.asarray(pos) + cls.STANCE_FOOT_POSITIONS[leg]
        while True:
            px, py, pz = pos  # pz must lower than shoulder-length
            px2, py2, pz2 = pos2 = pos ** 2
            stretch_len = math.sqrt(pos2.sum() - shoulder_len ** 2)
            try:
                hip_angle = 2 * math.atan((pz + math.sqrt(py2 + pz2 - shoulder_len ** 2)) /
                                          (py - shoulder_len))
                stretch_angle = -math.asin(px / stretch_len)
                shank_angle = math.acos((shank_len ** 2 + thigh_len ** 2 - stretch_len ** 2) /
                                        (2 * shank_len * thigh_len)) - math.pi
                thigh_angle = stretch_angle - math.asin(shank_len * math.sin(shank_angle) / stretch_len)
                return np.array((hip_angle, thigh_angle, shank_angle))
            except ValueError:
                pos *= 0.95

    @classmethod
    def forward_kinematics(cls, leg: int, angles: ARRAY_LIKE) -> tf.Odometry:
        """Calculate the position and orientation of the end-effector (foot) in BASE frame"""

        def _mdh_matrix(alpha, a, d, theta):
            ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
            return tf.Odometry(((ct, -st, 0),
                                (st * ca, ct * ca, -sa),
                                (st * sa, ct * sa, ca)),
                               (a, -sa * d, ca * d))

        shoulder_len, thigh_len, shank_len = cls.LINK_LENGTHS
        if leg % 2 == 1:
            shoulder_len *= -1
        a1, a2, a3 = angles
        tfm = (tf.Odometry(((0, 0, 1),
                            (-1, 0, 0),
                            (0, -1, 0)),
                           cls.HIP_OFFSETS[leg]) @
               _mdh_matrix(0, 0, 0, a1) @
               _mdh_matrix(0, shoulder_len, 0, np.pi / 2) @
               _mdh_matrix(-np.pi / 2, 0, 0, a2) @
               _mdh_matrix(0, thigh_len, 0, a3) @
               _mdh_matrix(0, shank_len, 0, 0) @
               tf.Odometry(((0, 0, -1),
                            (-1, 0, 0),
                            (0, 1, 0))))
        return tfm

    @classmethod
    def endeff_position(cls, leg: int, angles: ARRAY_LIKE) -> np.ndarray:
        """Calculate the position and orientation of the end-effector (foot) in HIP frame"""

        l1, l2, l3 = cls.LINK_LENGTHS
        if leg % 2 == 1:
            l1 *= -1
        a1, a2, a3 = angles
        s1, c1 = math.sin(a1), math.cos(a1)
        s2, c2 = math.sin(a2), math.cos(a2)
        s23, c23 = math.sin(a2 + a3), math.cos(a2 + a3)
        return np.array((
            (-l3 * s23 - l2 * s2),
            (l3 * s1 * c23 + l2 * s1 * c2 - l1 * c1),
            (-l3 * c1 * c23 - l1 * s1 - l2 * c1 * c2)
        ))
