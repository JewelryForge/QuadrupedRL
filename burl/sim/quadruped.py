from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
from collections import namedtuple, deque

from burl.utils.bc import Observable
from burl.sim.sensors import OrientationSensor, ContactStateSensor, OrientationRpySensor, GravityVectorSensor
from burl.utils.transforms import Rpy
from burl.utils import normalize, unit, JointInfo, make_cls
from burl.sim.motor import MotorSim, MotorMode

# TODO: DO NOT USE NAMEDTUPLE!!!
JointState = namedtuple('JointState', ('pos', 'vel', 'reaction_force', 'torque'),
                        defaults=(0., 0., (0., 0., 0., 0., 0., 0.), 0.))
ObservationRaw = namedtuple('ObservationRaw', ('joint_states', 'base_state', 'contact_states'))
Pose = namedtuple('Pose', ('position', 'orientation'), defaults=((0, 0, 0), (0, 0, 0, 1)))
Twist = namedtuple('Twist', ('linear', 'angular'), defaults=((0, 0, 0), (0, 0, 0)))

ROBOT = 'a1'


class QuadrupedSim(Observable):
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
        HIP_OFFSETS = np.array(((0.2399, -0.051, 0), (0.2399, 0.051, 0),
                                (-0.2399, -0.051, 0), (-0.2399, 0.051, 0)))
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
        COM_OFFSET = -np.array((0.012731, 0.002186, 0.000515))
        HIP_OFFSETS = np.array(((0.183, -0.047, 0.), (0.183, 0.047, 0.),
                                (-0.183, -0.047, 0.), (-0.183, 0.047, 0.)))
        STANCE_POSTURE = (0, 0.723, -1.445)
        TORQUE_LIMITS = (-33.5,) * 12, (33.5,) * 12

    ALLOWED_SENSORS = {ContactStateSensor: 12, OrientationSensor: 4, OrientationRpySensor: 3,
                       GravityVectorSensor: 3, }

    def __init__(self, sim_env=pybullet, **kwargs):
        self._env = sim_env
        self._urdf_file: str = kwargs.get('_urdf_file', self.URDF_FILE)
        self._self_collision_enabled: bool = kwargs.get('self_collision_enabled', False)
        self._frequency = kwargs.get('frequency', 240)
        _make_sensors = kwargs.get('make_sensors', ())
        super(QuadrupedSim, self).__init__(make_sensors=_make_sensors)
        _make_motor: Callable = kwargs.get('make_motor', MotorSim)
        self._motor: MotorSim = _make_motor(self, num=12, frequency=self._frequency)
        self._subordinates.append(self._motor)

        # print('dim', self._motor.observation_dim)
        self._latency: float = kwargs.get('latency', 0)
        # self._sensors = [m(self) for m in _make_sensors]
        # self._noisy: bool = kwargs.get('noisy', False)
        self._on_rack: bool = kwargs.get('on_rack', False)
        assert self._latency >= 0  # and self._time_step > 0

        # self._observation_dim = sum(s.observation_dim for s in self._sensors)
        self._joint_states: list[JointState] = []
        self._base_pose: Pose = Pose()
        self._base_twist: Twist = Twist()
        self._base_twist_in_base_frame: Twist = Twist()
        self._contact_states: tuple = (False,) * 13
        self._privileged_info = None
        self._step_counter: int = 0

        self._latency_steps = int(self._latency * self._frequency)
        if self._on_rack:
            # self._quadruped: int = self._load_robot_urdf()
            self._quadruped: int = self._load_robot_urdf(self.INIT_RACK_POSITION)
        else:
            self._quadruped: int = self._load_robot_urdf()
        self._num_joints = self._env.getNumJoints(self._quadruped)
        self._joint_names = [self.BASE_NAME]
        self._joint_names.extend('_'.join((l, j, s)) for l in self.LEG_NAMES
                                 for j, s in zip(self.JOINT_TYPES, self.JOINT_SUFFIX))

        self._joint_ids: list = self._init_joint_ids()
        self._motor_ids: list = [self._get_joint_id(l, j) for l in range(4) for j in range(3)]
        self._foot_ids: list = [self._get_joint_id(l, -1) for l in range(4)]
        self._observation_history = deque(maxlen=100)  # previous observation for RL is not preserved here
        self._command_history = deque(maxlen=100)
        self._init_posture()
        if self._on_rack:
            self._create_rack_constraint()
        self.set_physics_parameters()

    @property
    def robot_id(self):
        return self._quadruped

    @property
    def frequency(self):
        return self._frequency

    @property
    def action_limits(self):
        if self._motor.mode == MotorMode.TORQUE:
            return self.TORQUE_LIMITS
        elif self._motor.mode == MotorMode.POSITION:
            return tuple(zip(*(JointInfo(self._env.getJointInfo(self._quadruped, m)).limits for m in self._motor_ids)))

    def set_physics_parameters(self, **kwargs):  # TODO complete kwargs
        # for m in self._motor_ids:
        #     self._env.changeDynamics(self, -1, linearDamping=0, angularDamping=0)

        for f in self._foot_ids:
            self._env.changeDynamics(self._quadruped, f, spinningFriction=self.FOOT_SPINNING_FRICTION,
                                     lateralFriction=self.FOOT_LATERAL_FRICTION)
        self._env.setPhysicsEngineParameter(enableConeFriction=0)
        self._env.setJointMotorControlArray(self._quadruped, self._motor_ids, self._env.VELOCITY_CONTROL,
                                            forces=[self.JOINT_FRICTION] * len(self._motor_ids))
        for leg in self.LEG_NAMES:
            self._env.enableJointForceTorqueSensor(self._quadruped, self._get_joint_id(leg, -1))

    def reset(self, at_current_state=True):
        self._step_counter = 0
        self._observation_history.clear()
        self._command_history.clear()
        if self._on_rack:
            self._env.resetBasePositionAndOrientation(self._quadruped, self.INIT_RACK_POSITION,
                                                      self.orientation)
            self._env.resetBaseVelocity(self._quadruped, [0, 0, 0], [0, 0, 0])
        elif at_current_state:
            x, y, _ = self.position
            z = self.INIT_POSITION[2]
            _, _, yaw = self._env.getEulerFromQuaternion(self.orientation)
            orn_q = self._env.getQuaternionFromEuler([0.0, 0.0, yaw])
            self._env.resetBasePositionAndOrientation(self._quadruped, [x, y, z], orn_q)
            self._env.resetBaseVelocity(self._quadruped, [0, 0, 0], [0, 0, 0])
        else:
            self._env.resetBasePositionAndOrientation(self._quadruped, self.INIT_POSITION, (0, 0, 0, 0))
            self._env.resetBaseVelocity(self._quadruped, (0, 0, 0), (0, 0, 0))
        self._init_posture()
        self._motor.reset()
        self.update_observation()

    def random_dynamics(self, dynamics_parameters):
        pass

    def _on_update_observation(self):
        self._step_counter += 1
        joint_states = [JointState(*js) for js in self._env.getJointStates(self._quadruped, range(self._num_joints))]
        base_pose = Pose(*self._env.getBasePositionAndOrientation(self._quadruped))
        base_twist = Twist(*self._env.getBaseVelocity(self._quadruped))
        contact_states = self._get_contact_states()
        observation_current = ObservationRaw(joint_states, (base_pose, base_twist), contact_states)
        # print(contact_states)
        self._observation_history.append(observation_current)
        observation = self._get_observation()
        self._joint_states = observation.joint_states
        self._base_pose, self._base_twist = observation.base_state
        # NOTICE: here 'base_twist_in_base_frame' should not be influenced by the noise of base pose
        # So the quantities here shouldn't be noisy
        self._base_twist_in_base_frame = Twist(*self._transform_world2base(*self._base_twist))
        self._contact_states = observation.contact_states
        self._privileged_info = observation_current

    def is_safe(self):
        r, p, _ = Rpy.from_quaternion(self.orientation)
        return abs(r) < np.pi / 4 and abs(p) < np.pi / 4

    @property
    def privileged_info(self) -> None | ObservationRaw:
        return self._privileged_info

    @property
    def position(self):
        return self._base_pose.position

    @property
    def orientation(self):
        return self._base_pose.orientation

    # def sensor_interface(self, quantity_name):
    #     if quantity_name ==
    #     quantity_interface_dict = {
    #
    #     }
    #     return

    def get_pose(self):
        return self._base_pose

    def get_twist(self):
        return self._base_twist_in_base_frame

    def get_contact_states(self):
        return self._contact_states

    def get_joint_states(self):
        return (self._joint_states[m] for m in self._motor_ids)

    def get_joint_position_error_history(self, moment):
        assert moment < 0
        num_past_steps = int((self._latency - moment) * self._frequency)
        idx = 0 if len(self._observation_history) <= num_past_steps else -1 - num_past_steps
        joint_positions = [self._observation_history[idx].joint_states[m].pos for m in self._motor_ids]
        return self._command_history[idx] - np.array(joint_positions)

    def get_joint_velocity_history(self, moment):
        assert moment < 0
        num_past_steps = int((self._latency - moment) * self._frequency)
        idx = 0 if len(self._observation_history) <= num_past_steps else -1 - num_past_steps
        joint_velocities = [self._observation_history[idx].joint_states[m].vel for m in self._motor_ids]
        return joint_velocities

    # [1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    def apply_command(self, motor_commands):
        motor_commands = np.asarray(motor_commands)
        self._command_history.append(motor_commands)
        torques = self._motor.set_command(motor_commands)
        self._env.setJointMotorControlArray(self._quadruped, self._motor_ids,
                                            self._env.TORQUE_CONTROL, forces=torques)
        return torques

    def add_disturbance_on_base(self, force, pos=(0, 0, 0)):
        self._env.applyExternalForce(self._quadruped, -1, force, pos)

    def ik(self, leg: int | str, pos, frame='base'):
        assert ROBOT == 'a1'
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        shoulder_length, thigh_length, shank_length = self.LINK_LENGTHS
        # print(leg)
        if self.LEG_NAMES[leg].endswith('R'):
            shoulder_length *= -1

        def _ik_hip_frame(_pos):
            dx, dy, dz = _pos
            distance = np.linalg.norm(_pos)
            hip_angle_bias = np.arctan2(dy, dz)
            _sum = np.arcsin(distance * shoulder_length / np.hypot(dy, dz) / distance)
            opt1, opt2 = normalize(_sum - hip_angle_bias), normalize(np.pi - _sum - hip_angle_bias)
            hip_angle = opt1 if abs(opt1) < abs(opt2) else opt2
            shoulder_vector = np.array((0, np.cos(hip_angle), np.sin(hip_angle))) * shoulder_length
            foot_position_shoulder = _pos - shoulder_vector
            foot_distance_shoulder = np.linalg.norm(foot_position_shoulder)
            thigh_length_2, shank_length_2, foot_distance_shoulder_2 = \
                thigh_length ** 2, shank_length ** 2, foot_distance_shoulder ** 2
            angle_shank = np.arccos((thigh_length_2 + shank_length_2 - foot_distance_shoulder_2)
                                    / (2 * thigh_length * shank_length)) - np.pi
            angle_thigh = np.arccos((thigh_length_2 + foot_distance_shoulder_2 - shank_length_2)
                                    / (2 * thigh_length * foot_distance_shoulder))
            normal = np.cross(shoulder_vector, np.cross((0, 0, -1), shoulder_vector))
            angle_thigh -= np.arccos(np.dot(unit(normal), unit(foot_position_shoulder))) * np.sign(dx)
            # print(np.dot(unit(normal), unit(foot_distance_shoulder)), angle_thigh)
            return hip_angle, normalize(angle_thigh), normalize(angle_shank)

        pos = np.asarray(pos)
        if frame == 'world':
            return _ik_hip_frame(self._transform_world2base(pos) - self.HIP_OFFSETS[leg])
        if frame == 'base':
            return _ik_hip_frame(pos - self.HIP_OFFSETS[leg])
        if frame == 'hip':
            return _ik_hip_frame(pos)
        if frame == 'shoulder':
            return _ik_hip_frame(pos + (0, shoulder_length, 0))
        raise RuntimeError(f'Unknown Frame named {frame}')

    def fk(self, leg):
        pass

    def print_joint_info(self):
        for i in range(p.getNumJoints(self._quadruped)):
            print(JointInfo(p.getJointInfo(self._quadruped, i)))

    def _load_robot_urdf(self, pos=None, orn=None):
        if self._self_collision_enabled:
            return self._env.loadURDF(
                self._urdf_file,
                pos if pos else self.INIT_POSITION,
                orn if orn else self.INIT_ORIENTATION,
                flags=self._env.URDF_USE_SELF_COLLISION)
        else:
            return self._env.loadURDF(
                self._urdf_file,
                pos if pos else self.INIT_POSITION,
                orn if orn else self.INIT_ORIENTATION)

    def _init_posture(self):
        for i in range(12):
            self._env.resetJointState(self._quadruped, self._motor_ids[i], self.STANCE_POSTURE[i % 3])

    def _init_joint_ids(self):
        joint_name_to_id = {}
        for i in range(self._env.getNumJoints(self._quadruped)):
            joint_info = self._env.getJointInfo(self._quadruped, i)
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
        return self._joint_ids[1 + leg * 4 + joint_type]

    def _create_rack_constraint(self):
        self._env.createConstraint(self._quadruped, -1,
                                   childBodyUniqueId=-1,
                                   childLinkIndex=-1,
                                   jointType=self._env.JOINT_FIXED,
                                   jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=self.INIT_RACK_POSITION,
                                   childFrameOrientation=self.INIT_ORIENTATION)

    def _get_observation(self) -> ObservationRaw:
        if len(self._observation_history) <= self._latency_steps:
            return self._observation_history[0]
        else:
            return self._observation_history[-(self._latency_steps + 1)]

    def _get_contact_states(self):
        def _get_contact_state(link_id):
            return bool(self._env.getContactPoints(bodyA=self._quadruped, linkIndexA=link_id))

        base_contact = _get_contact_state(0)
        contact_states = []
        for leg in range(4):  # FIXME: CONTACT JUDGEMENT SHOULD BE THOUGHT OVER
            base_contact = base_contact or _get_contact_state(leg * 5 + 1) or _get_contact_state(leg * 5 + 2)
            contact_states.extend(_get_contact_state(leg * 5 + i) for i in range(3, 6))
        contact_states.insert(0, base_contact)
        # print(contact_states)
        return contact_states

    def _transform_world2base(self, *vectors):
        def _transform_once(vec):
            _, orientation_inverse = self._env.invertTransform((0, 0, 0), self.orientation)
            relative_vec, _ = self._env.multiplyTransforms((0, 0, 0), orientation_inverse,
                                                           vec, (0, 0, 0, 1))
            return relative_vec

        if len(vectors) == 1:
            return _transform_once(vectors[0])
        return (_transform_once(v) for v in vectors)


if __name__ == '__main__':
    from burl.sim.sensors import MotorEncoder

    np.set_printoptions(precision=3, linewidth=1000)
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    make_motor = make_cls(MotorSim, make_sensors=[MotorEncoder])
    q = QuadrupedSim(pybullet_client=p, make_motor=make_motor, on_rack=False)
    print('dim', q.observation_dim)
    # q.print_joint_info()
    p.setGravity(0, 0, -9.8)
    # c = p.loadURDF("cube.urdf", globalScaling=0.1)
    print(q.action_limits)
    for _ in range(100000):
        # if _ % 1000 == 0:
        #     q.reset()
        p.stepSimulation()
        q.update_observation()
        # print(q.step())
        time.sleep(1. / 240.)
        cmd0 = q.ik(0, (0, -0.08505, -0.3), 'hip')
        cmd1 = q.ik(1, (0, 0.08505, -0.3), 'hip')
        cmd2 = q.ik(2, (0, -0.08505, -0.3), 'hip')
        cmd3 = q.ik(3, (0, 0.08505, -0.3), 'hip')
        # print('cmd', [cmd0, cmd1, cmd2, cmd3])
        tq = q.apply_command(np.concatenate([cmd0, cmd1, cmd2, cmd3]))
