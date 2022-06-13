import collections
import copy
import math
import os
from typing import Optional, Deque

import dm_control.mujoco as mjlib
import numpy as np
from dm_control import mjcf, composer
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base

from qdpgym import tf, utils as ut
from qdpgym.sim import rsc_dir
from qdpgym.sim.abc import Command, Snapshot
from qdpgym.sim.abc import Quadruped, QuadrupedHandle, ARRAY_LIKE
from qdpgym.sim.common.motor import PdMotorSim, ActuatorNetSim
from qdpgym.sim.common.noisyhandle import NoisyHandle
from .terrain import Arena


class AliengoObservableMj(composer.Observables):
    @composer.observable
    def position(self):
        return observable.MJCFFeature('xpos', self._entity.root_body)

    @composer.observable
    def orientation(self):
        return observable.MJCFFeature('xquat', self._entity.root_body,
                                      corruptor=lambda v, random_state: tf.Quaternion.from_wxyz(v))

    @composer.observable
    def rotation(self):
        return observable.MJCFFeature('xmat', self._entity.root_body,
                                      corruptor=lambda v, random_state: v.reshape(3, 3))

    @composer.observable
    def rpy(self):
        def q2rpy(q):
            return tf.Rpy.from_quaternion(tf.Quaternion.from_wxyz(q))

        return observable.MJCFFeature('xquat', self._entity.root_body,
                                      corruptor=lambda v, random_state: q2rpy(v))

    @composer.observable
    def linear_vel(self):
        return observable.MJCFFeature('cvel', self._entity.root_body, index=slice(3, None))

    @composer.observable
    def angular_vel(self):
        return observable.MJCFFeature('cvel', self._entity.root_body, index=slice(3))

    @composer.observable
    def joint_pos(self):
        return observable.MJCFFeature('qpos', self._entity.observable_joints)

    @composer.observable
    def joint_vel(self):
        return observable.MJCFFeature('qvel', self._entity.observable_joints)

    @composer.observable
    def sensors_velocimeter(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.mjcf_model.sensor.velocimeter)

    @composer.observable
    def sensors_gyro(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.mjcf_model.sensor.gyro)

    @composer.observable
    def sensors_accelerometer(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.mjcf_model.sensor.accelerometer)

    @composer.observable
    def foot_pos(self):
        return observable.MJCFFeature('xpos', self._entity.end_effectors)


class AliengoModelMj(base.Walker):
    MODEL_PATH = 'aliengo/model/aliengo.xml'

    def _build(self, name='aliengo'):
        self._mjcf_raw = mjcf.from_path(os.path.join(rsc_dir, self.MODEL_PATH))
        self._mjcf_root = copy.deepcopy(self._mjcf_raw)
        if name:
            self._mjcf_root.model = name

        self._torso_geom_ids: Optional[list] = None
        self._leg_geom_ids: Optional[list] = None
        self._foot_geom_ids: Optional[list] = None

    def initialize_episode_mjcf(self, random_state=None):
        if random_state is None:
            return
        else:
            pass

    def after_compile(self, physics, random_state=None):
        super().after_compile(physics, random_state)

        def _get_entity_geoms(bodies):
            geoms = []
            for body in bodies:
                for geom in body.find_all('geom', immediate_children_only=True):
                    if geom.contype != 0:
                        geoms.append(physics.bind(geom).element_id)
            return geoms

        self._torso_geom_ids = _get_entity_geoms(self.unsafe_contact_bodies)
        self._leg_geom_ids = _get_entity_geoms(self.leg_bodies)
        self._foot_geom_ids = _get_entity_geoms(self._foot_bodies)

    def collect_contacts(self, physics):
        torso_contact = False
        leg_contacts = [0] * 12
        foot_contact_forces = np.zeros((4, 3))
        force_torque = np.zeros(6)
        for i, contact in enumerate(physics.data.contact):
            geom1, geom2 = contact.geom1, contact.geom2,
            is_valid = contact.dist < contact.includemargin
            if geom1 == 0 and is_valid:
                if geom2 in self._torso_geom_ids:
                    torso_contact = True
                elif geom2 in self._leg_geom_ids:
                    leg_contacts[self._leg_geom_ids.index(contact.geom2)] = 1
                    if contact.geom2 in self._foot_geom_ids:
                        mjlib.mj_contactForce(physics.model.ptr, physics.data.ptr, i, force_torque)
                        idx = self._foot_geom_ids.index(contact.geom2)
                        foot_contact_forces[idx] = contact.frame.reshape(3, 3).T @ force_torque[:3]

        return torso_contact, leg_contacts, foot_contact_forces

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @composer.cached_property
    def actuators(self):
        return self._mjcf_root.find_all('actuator')

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'torso')

    @composer.cached_property
    def unsafe_contact_bodies(self):
        return (self._mjcf_root.find('body', 'torso'),
                self._mjcf_root.find('body', 'FR_hip'),
                self._mjcf_root.find('body', 'FL_hip'),
                self._mjcf_root.find('body', 'RR_hip'),
                self._mjcf_root.find('body', 'RL_hip'))

    @composer.cached_property
    def leg_bodies(self):
        return (self._mjcf_root.find('body', 'FR_thigh'),
                self._mjcf_root.find('body', 'FR_calf'),
                self._mjcf_root.find('body', 'FR_foot'),
                self._mjcf_root.find('body', 'FL_thigh'),
                self._mjcf_root.find('body', 'FL_calf'),
                self._mjcf_root.find('body', 'FL_foot'),
                self._mjcf_root.find('body', 'RR_thigh'),
                self._mjcf_root.find('body', 'RR_calf'),
                self._mjcf_root.find('body', 'RR_foot'),
                self._mjcf_root.find('body', 'RL_thigh'),
                self._mjcf_root.find('body', 'RL_calf'),
                self._mjcf_root.find('body', 'RL_foot'))

    @composer.cached_property
    def entity_bodies(self):
        return self.unsafe_contact_bodies + self.leg_bodies

    @property
    def _foot_bodies(self):
        return (self._mjcf_root.find('body', 'FR_foot'),
                self._mjcf_root.find('body', 'FL_foot'),
                self._mjcf_root.find('body', 'RR_foot'),
                self._mjcf_root.find('body', 'RL_foot'))

    @composer.cached_property
    def end_effectors(self):
        return self._foot_bodies

    @composer.cached_property
    def observable_joints(self):
        return [actuator.joint for actuator in self.actuators]

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    @composer.cached_property
    def ground_contact_geoms(self):
        foot_geoms = []
        for foot in self._foot_bodies:
            foot_geoms.extend(foot.find_all('geom'))
        return tuple(foot_geoms)

    def _build_observables(self):
        return AliengoObservableMj(self)


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

    def __init__(self, frequency: int, motor: str, noisy=True):
        self._freq = frequency
        if motor == 'pd':
            self._motor = PdMotorSim(self._freq, 150, 4)
        elif motor == 'actuator_net':
            self._motor = ActuatorNetSim(self._freq)
            self._motor.load_params(os.path.join(rsc_dir, 'acnet_220526.pt'))
        else:
            raise NotImplementedError
        self._noisy_on = noisy

        self._entity = AliengoModelMj()
        self._handle: AliengoObservableMj = self._entity.observables
        self._motor.set_joint_limits(*zip(*self.JOINT_LIMITS))
        self._motor.set_torque_limits(self.TORQUE_LIMITS)
        self._physics: Optional[mjcf.Physics] = None
        self._noisy: Optional[NoisyHandle] = None
        if self._noisy_on:
            self._noisy = NoisyHandle(self, frequency)

        self._state: Optional[Snapshot] = None
        self._state_history: Deque[Snapshot] = collections.deque(maxlen=100)
        self._cmd: Optional[Command] = None
        self._cmd_history: Deque[Command] = collections.deque(maxlen=100)

        self._random_dynamics = False
        self._latency_range = None

        self._init_pose = (0., 0., 0.)  # x, y, yaw

    @property
    def noisy(self) -> QuadrupedHandle:
        return self._noisy if self._noisy_on else self

    @property
    def entity(self):
        return self._entity

    @property
    def handle(self):
        return self._handle

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

        self._init_pose = (x, y, yaw)

    def add_to(self, arena: Arena):
        if self._entity.parent is arena:
            attached = mjcf.get_attachment_frame(self._entity.mjcf_model)
        else:
            if arena.parent is not None:
                self._entity.detach()
            attached = arena.attach(self._entity)
            self._entity.create_root_joints(attached)

        x, y, yaw = self._init_pose
        foot_xy = (np.array(self.STANCE_FOOT_POSITIONS) + np.array(self.HIP_OFFSETS))[:, :2]
        if yaw != 0.:
            cy, sy = np.cos(yaw), np.sin(yaw)
            trans = np.array(((cy, -sy),
                              (sy, cy)))
            foot_xy = np.array([trans @ f_xy for f_xy in foot_xy])
        else:
            cy, sy = 1., 0.
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

        attached.pos = (x, y, init_height)
        attached.quat = orn

    def init_mjcf_model(self, random_state):
        if self._random_dynamics:
            self._entity.initialize_episode_mjcf(random_state)

    def init_physics(self, physics, random_state: np.random.RandomState, cfg=None):
        self._physics = physics
        self._entity.after_compile(physics)
        if cfg is None:
            cfg = self.STANCE_CONFIG
        self._entity.configure_joints(physics, cfg)
        if self._noisy_on and self._latency_range is not None:
            self._noisy.latency = random_state.uniform(*self._latency_range)

    def update_observation(self, random_state, minimal=False):
        """Get robot states from mujoco"""
        if minimal:
            # Only update motor observations, for basic locomotion
            self._motor.update_observation(self._handle.joint_pos(self._physics),
                                           self._handle.joint_vel(self._physics))
            return

        contact_info = self._entity.collect_contacts(self._physics)
        self._state = Snapshot(
            position=self._handle.position(self._physics),
            orientation=self._handle.orientation(self._physics),
            rotation=self._handle.rotation(self._physics),
            rpy=self._handle.rpy(self._physics),
            linear_vel=self._handle.linear_vel(self._physics),
            angular_vel=self._handle.angular_vel(self._physics),
            joint_pos=self._handle.joint_pos(self._physics),
            joint_vel=self._handle.joint_vel(self._physics),
            foot_pos=self._handle.foot_pos(self._physics),
            velocimeter=self._handle.sensors_velocimeter(self._physics),
            gyro=self._handle.sensors_gyro(self._physics),
            accelerometer=self._handle.sensors_accelerometer(self._physics),
            torso_contact=contact_info[0],
            leg_contacts=np.array(contact_info[1]),
            contact_forces=contact_info[2]
        )
        self._state.rpy_rate = tf.get_rpy_rate_from_ang_vel(self._state.rpy, self._state.angular_vel)
        self._state.force_sensor = np.array(
            [self._state.rotation.T @ force for force in self._state.contact_forces]
        )
        if self._noisy_on:
            self._noisy.update_observation(self._state, random_state)
        self._state_history.append(self._state)

        # self._motor.update_observation(self._state.joint_pos, self._state.joint_vel)
        self._motor.update_observation(self.noisy.get_joint_pos(), self.noisy.get_joint_vel())

    def apply_command(self, motor_commands: ARRAY_LIKE):
        """
        Calculate desired joint torques of position commands with motor model.
        :param motor_commands: array of desired motor positions.
        :return: array of desired motor torques.
        """
        motor_commands = np.asarray(motor_commands)
        torques = self._motor.apply_position(motor_commands)
        self._entity.apply_action(self._physics, torques, None)
        self._cmd = Command(
            command=motor_commands,
            torque=torques
        )
        self._cmd_history.append(self._cmd)
        return torques

    def apply_torques(self, torques):
        self._entity.apply_action(self._physics, torques, None)
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

    def get_joint_pos(self) -> np.ndarray:
        return self._state.joint_pos

    def get_joint_vel(self) -> np.ndarray:
        return self._state.joint_vel

    def get_last_command(self) -> np.ndarray:
        return self._cmd.command if self._cmd is not None else None

    def get_last_torque(self) -> np.ndarray:
        return self._cmd.torque if self._cmd is not None else None

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
