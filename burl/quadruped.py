from __future__ import annotations

import time
from abc import ABC
from typing import Callable

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
from collections import namedtuple, deque

from burl.utils import analyse_joint_info
from burl.motor import MotorSim

JointState = namedtuple('JointState', ('pos', 'vel', 'reaction_force', 'torque'),
                        defaults=(0., 0., (0., 0., 0., 0., 0., 0.), 0.))
ObservationRaw = namedtuple('ObservationRaw', ('joint_states', 'base_state', 'contact_states'))
Pose = namedtuple('Pose', ('position', 'orientation'), defaults=((0, 0, 0), (0, 0, 0, 1)))
Twist = namedtuple('Twist', ('linear', 'angular'), defaults=((0, 0, 0), (0, 0, 0)))

ROBOT = 'a1'


def normalize(x):
    return x - ((x + np.pi) // (2 * np.pi)) * (2 * np.pi)


class QuadrupedBase(ABC):
    def __init__(self):
        pass

    def reset(self, *args):
        pass

    def step(self):
        pass

    def update_observation(self):
        pass

    def apply_command(self, motor_commands):
        pass

    def ik(self, *args):
        pass


class QuadrupedSim(QuadrupedBase):
    INIT_POSITION = [0, 0, .33]
    INIT_RACK_POSITION = [0, 0, 1]
    INIT_ORIENTATION = [0, 0, 0, 1]
    NUM_MOTORS = 12
    if ROBOT == 'aliengo':
        LEG_NAMES = ['FR', 'FL', 'RR', 'RL']
        JOINT_TYPES = ['hip', 'thigh', 'calf', 'foot']
        JOINT_SUFFIX = ['joint', 'joint', 'joint', 'fixed']
        BASE_NAME = 'floating_base'
        URDF_FILE = "/home/jewel/Workspaces/teacher-student/urdf/aliengo/xacro/aliengo.urdf"
        COM_OFFSET = np.array((0.008465, 0.004045, -0.000763))
        HIP_OFFSETS = np.array(((0.2399, -0.051, 0), (0.2399, 0.051, 0), (-0.2399, -0.051, 0), (-0.2399, 0.051, 0)))
    elif ROBOT == 'a1':
        BASE_NAME = 'imu_joint'
        LEG_NAMES = ['FR', 'FL', 'RR', 'RL']
        JOINT_TYPES = ['hip', 'upper', 'lower', 'toe']
        JOINT_SUFFIX = ['joint', 'joint', 'joint', 'fixed']
        URDF_FILE = '/home/jewel/Workspaces/teacher-student/urdf/a1/a1.urdf'
        LINK_LENGTHS = [0.08505, 0.2, 0.2]
        JOINT_FRICTION = 0.025
        FOOT_LATERAL_FRICTION = 1.0
        FOOT_SPINNING_FRICTION = 0.2
        FOOT_RESTITUTION = 0.3
        COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
        HIP_OFFSETS = np.array(((0.183, -0.047, 0.), (0.183, 0.047, 0.),
                                (-0.183, -0.047, 0.), (-0.183, 0.047, 0.)))

    def __init__(self, **kwargs):
        super(QuadrupedSim, self).__init__()
        self._urdf_file: str = kwargs.get('_urdf_file', self.URDF_FILE)
        self._bullet = kwargs.get('pybullet_client', pybullet)
        self._self_collision_enabled: bool = kwargs.get('self_collision_enabled', False)
        self._render_enabled: bool = kwargs.get('_render_enabled', True)
        make_motor: Callable = kwargs.get('make_motor', MotorSim)
        self._motor: MotorSim = make_motor()
        self._latency: float = kwargs.get('latency', 0)
        self._noisy: bool = kwargs.get('noisy', False)
        self._on_rack: bool = kwargs.get('on_rack', False)
        assert self._latency >= 0  # and self._time_step > 0

        # self._bullet.setTimeStep(self._time_step)
        self._frequency = self._motor.frequency
        self._joint_states: list[JointState] = []
        self._base_pose: Pose = Pose()
        self._base_twist: Twist = Twist()
        self._base_twist_in_base_frame: Twist = Twist()
        self._contact_states: tuple = (False,) * 13
        self._step_counter: int = 0

        self._latency_steps = int(self._latency * self._frequency)
        if self._on_rack:
            self._quadruped: int = self._load_robot_urdf(self.INIT_RACK_POSITION)
        else:
            self._quadruped: int = self._load_robot_urdf()
        self._num_joints = self._bullet.getNumJoints(self._quadruped)
        self._joint_names = [self.BASE_NAME]
        self._joint_names.extend('_'.join((l, j, s)) for l in self.LEG_NAMES
                                 for j, s in zip(self.JOINT_TYPES, self.JOINT_SUFFIX))

        self._joint_ids: list = self._init_joint_ids()
        self._motor_ids: list = [self._get_joint_id(l, j) for l in range(4) for j in range(3)]
        self._foot_ids: list = [self._get_joint_id(l, -1) for l in range(4)]
        self._observation_history = deque(maxlen=100)  # previous observation for RL is not preserved here

        if self._on_rack:
            self._create_rack_constraint()
        self.set_physics_parameters()

    @property
    def robot_id(self):
        return self._quadruped

    @property
    def frequency(self):
        return self._frequency

    def set_physics_parameters(self, **kwargs):  # TODO
        for i in range(4):
            self._bullet.resetJointState(self._quadruped, self._motor_ids[i * 3], 0)
            self._bullet.resetJointState(self._quadruped, self._motor_ids[i * 3 + 1], 0.723)
            self._bullet.resetJointState(self._quadruped, self._motor_ids[i * 3 + 2], -1.445)
        # self._bullet.stepSimulation()

        # for m in self._motor_ids:
        #     self._bullet.changeDynamics(self, -1, linearDamping=0, angularDamping=0)
        for f in self._foot_ids:
            self._bullet.changeDynamics(self._quadruped, f, spinningFriction=self.FOOT_SPINNING_FRICTION,
                                        lateralFriction=self.FOOT_LATERAL_FRICTION)
        self._bullet.setPhysicsEngineParameter(enableConeFriction=0)
        self._bullet.setJointMotorControlArray(self._quadruped, self._motor_ids, self._bullet.VELOCITY_CONTROL,
                                               forces=[self.JOINT_FRICTION] * len(self._motor_ids))
        for leg in self.LEG_NAMES:
            self._bullet.enableJointForceTorqueSensor(self._quadruped, self._get_joint_id(leg, -1))

    def reset(self, at_current_state=True):
        self._step_counter = 0
        self._observation_history.clear()
        if at_current_state:
            self._reset_at_current_state()
        else:
            self._bullet.resetBasePositionAndOrientation(self._quadruped, (0, 0, 0), (0, 0, 0, 0))
            self._bullet.resetBaseVelocity(self._quadruped, (0, 0, 0), (0, 0, 0))
        self._motor.reset()

    def step(self):
        self._bullet.stepSimulation()
        return self.update_observation()

    def update_observation(self):
        self._step_counter += 1
        joint_states = [JointState(*js) for js in self._bullet.getJointStates(self._quadruped, range(self._num_joints))]
        base_pose = Pose(*self._bullet.getBasePositionAndOrientation(self._quadruped))
        base_twist = Twist(*self._bullet.getBaseVelocity(self._quadruped))
        contact_states = self._get_contact_states()
        self._observation_history.append(ObservationRaw(joint_states, (base_pose, base_twist), contact_states))
        observation = self._get_observation()
        self._joint_states = observation.joint_states
        self._base_pose, self._base_twist = observation.base_state
        self._base_twist_in_base_frame = Twist(*self._transform_world2base(*self._base_twist))
        self._contact_states = observation.contact_states
        # print('motor_ids', self._motor_ids)
        # motor_pos = [self._joint_states[m].pos for m in self._motor_ids]
        motor_pos = [self._joint_states[m].pos for m in self._motor_ids]
        self._motor.update_observation(motor_pos)
        # print('motor_pos', *[f'{p:.3f}' for p in motor_pos])
        return observation

    # [1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    def apply_command(self, motor_commands):
        torques = self._motor.set_command(motor_commands)
        # print(torques)
        self._bullet.setJointMotorControlArray(self._quadruped, self._motor_ids,
                                               self._bullet.TORQUE_CONTROL, forces=torques)
        return torques

    def add_disturbance_on_base(self, force, pos=(0, 0, 0)):
        self._bullet.applyExternalForce(self._quadruped, -1, force, pos)

    def ik(self, leg: int | str, pos, frame='base'):
        assert ROBOT == 'a1'
        shoulder_length, thigh_length, shank_length = self.LINK_LENGTHS
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        if self.LEG_NAMES[leg].endswith('R'):
            shoulder_length *= -1

        pos = np.asarray(pos)
        if frame == 'world':
            pos = self._transform_world2base(pos)
            frame = 'base'
        if frame == 'base':
            pos -= self.HIP_OFFSETS[leg]
            frame = 'hip'
        if frame == 'hip':
            dx, dy, dz = pos
            distance = np.linalg.norm(pos)
            hip_angle_bias = np.arctan2(dy, dz)
            _sum = np.arcsin(distance * shoulder_length / np.hypot(dy, dz) / distance)
            opt1, opt2 = normalize(_sum - hip_angle_bias), normalize(np.pi - _sum - hip_angle_bias)
            hip_angle = opt1 if abs(opt1) < abs(opt2) else opt2
            shoulder_vector = np.array((0, np.cos(hip_angle), np.sin(hip_angle))) * shoulder_length
            foot_position_shoulder = pos - shoulder_vector
            foot_distance_shoulder = np.linalg.norm(foot_position_shoulder)
            thigh_length_2, shank_length_2, foot_distance_shoulder_2 = \
                thigh_length ** 2, shank_length ** 2, foot_distance_shoulder ** 2
            angle_shank = np.arccos((thigh_length_2 + shank_length_2 - foot_distance_shoulder_2)
                                    / (2 * thigh_length * shank_length)) - np.pi
            angle_thigh = np.arccos((thigh_length_2 + foot_distance_shoulder_2 - shank_length_2)
                                    / (2 * thigh_length * foot_distance_shoulder))
            normal = np.cross(shoulder_vector, np.cross((0, 0, -1), shoulder_vector))
            angle_thigh += np.arccos(np.dot(normal / np.linalg.norm(normal),
                                            foot_position_shoulder / np.linalg.norm(foot_position_shoulder)))
            return hip_angle, normalize(angle_thigh), normalize(angle_shank)

        raise RuntimeError(f'Unknown Frame named {frame}')

    def fk(self, leg):
        pass

    def print_joint_info(self):
        for i in range(p.getNumJoints(self._quadruped)):
            print(analyse_joint_info(p.getJointInfo(self._quadruped, i)))

    def _load_robot_urdf(self, pos=None, orn=None):
        if self._self_collision_enabled:
            return self._bullet.loadURDF(
                self._urdf_file,
                pos if pos else self.INIT_POSITION,
                orn if orn else self.INIT_ORIENTATION,
                flags=self._bullet.URDF_USE_SELF_COLLISION)
        else:
            return self._bullet.loadURDF(
                self._urdf_file,
                pos if pos else self.INIT_POSITION,
                orn if orn else self.INIT_ORIENTATION)

    def _init_joint_ids(self):
        joint_name_to_id = {}
        for i in range(self._bullet.getNumJoints(self._quadruped)):
            joint_info = self._bullet.getJointInfo(self._quadruped, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        return [joint_name_to_id.get(n, -1) for n in self._joint_names]

    def _get_joint_id(self, leg: int | str, joint_type: int | str = 0):
        if leg == -1 or leg == self.BASE_NAME:  # BASE
            return self._joint_ids[0]
        if isinstance(joint_type, str):
            joint_type = self.JOINT_TYPES.index(joint_type)
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        if joint_type < 0:
            joint_type += 4
        # print(leg, joint_type, self._joint_ids)
        return self._joint_ids[1 + leg * 4 + joint_type]

    def _create_rack_constraint(self):
        self._bullet.createConstraint(self._quadruped, -1,
                                      childBodyUniqueId=-1,
                                      childLinkIndex=-1,
                                      jointType=self._bullet.JOINT_FIXED,
                                      jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0],
                                      childFramePosition=self.INIT_RACK_POSITION,
                                      childFrameOrientation=self.INIT_ORIENTATION)

    def _get_observation(self):
        if len(self._observation_history) <= self._latency_steps:
            observation = self._observation_history[0]
        else:
            observation = self._observation_history[-(self._latency_steps + 1)]
        return self._add_observation_noise(observation) if self._noisy else observation

    def _get_contact_states(self):
        def _get_contact_state(link_id):
            return bool(self._bullet.getContactPoints(
                bodyA=self._quadruped, linkIndexA=link_id, bodyB=0, linkIndexB=-1))

        base_contact = _get_contact_state(0)
        contact_states = []
        for leg in range(4):  # FIXME: CONTACT JUDGEMENT SHOULD BE THOUGHT OVER
            base_contact = base_contact or _get_contact_state(leg * 5 + 1) or _get_contact_state(leg * 5 + 2)
            contact_states.extend(_get_contact_state(leg * 5 + i) for i in range(3, 6))
        contact_states.insert(0, base_contact)
        return contact_states

    def _transform_world2base(self, *vectors):
        def _transform_once(vec):
            _, orientation_inversed = self._bullet.invertTransform((0, 0, 0), self._base_pose.orientation)
            relative_vec, _ = self._bullet.multiplyTransforms((0, 0, 0), orientation_inversed,
                                                              vec, (0, 0, 0, 1))
            return relative_vec

        if len(vectors) == 1:
            return _transform_once(vectors[0])
        return (_transform_once(v) for v in vectors)

    def _reset_at_current_state(self):
        x, y, _ = self._base_pose.position
        z = self.INIT_POSITION[2]
        _, _, yaw = self._bullet.getEulerFromQuaternion(self._base_pose.orientation)
        orn_q = self._bullet.getQuaternionFromEuler([0.0, 0.0, yaw])
        self._bullet.resetBasePositionAndOrientation(self._quadruped, [x, y, z], orn_q)
        self._bullet.resetBaseVelocity(self._quadruped, [0, 0, 0], [0, 0, 0])

    def _add_observation_noise(self, data):
        return data  # TODO: think about how to get noisy observation properly


if __name__ == '__main__':
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    q = QuadrupedSim(pybullet_client=p, on_rack=False)

    p.setGravity(0, 0, -9.8)
    c = p.loadURDF("cube.urdf", globalScaling=0.1)

    q.print_joint_info()
    for _ in range(100000):
        q.step()
        time.sleep(1. / 240.)
        cmd0 = q.ik(0, (0, -0.08505, -0.3), 'hip')
        cmd1 = q.ik(1, (0, 0.08505, -0.3), 'hip')
        cmd2 = q.ik(2, (0, -0.08505, -0.3), 'hip')
        cmd3 = q.ik(3, (0, 0.08505, -0.3), 'hip')
        tq = q.apply_command(np.concatenate([cmd0, cmd1, cmd2, cmd3]))
