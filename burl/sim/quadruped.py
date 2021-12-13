from __future__ import annotations

import math
import os.path
from collections import deque
from typing import Deque

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from burl.rl.state import JointStates, Pose, Twist, ContactStates, ObservationRaw, BaseState, FootStates
from burl.sim.motor import MotorSim
from burl.utils import normalize, unit, JointInfo, make_cls, g_cfg, vec_cross
from burl.utils.transforms import Rpy, Rotation, Odometry, get_rpy_rate_from_angular_velocity

TP_ZERO3 = (0., 0., 0.)
TP_Q0 = (0., 0., 0., 1.)


class Quadruped(object):
    """
    A class for observing and controlling a quadruped in pybullet.

    For specific robot, the following attribute should be specified according to the urdf file.
    This class is not responsible for pybullet.stepSimulation, so stepSimulation yourself.
    Before a simulation cycle, run updateObservation() to set initial states.
    A general process is: applyCommand -> stepSimulation -> updateObservation.
    Some attribute like position represents some current real states; run methods like getJointHistory for history.
    Method updateObservation automatically saves real states and a noisy version with latency.
    Expect some commonly used attribute like position and orientation,
    it's suggested to use method with prefix 'get' to get observations.
    A 'get' method will give a real observation, unless there's a 'noisy' option and 'noisy=True' is specified.
    """

    INIT_POSITION: np.ndarray
    INIT_RACK_POSITION: np.ndarray
    INIT_ORIENTATION: np.ndarray
    NUM_MOTORS = 12
    LEG_NAMES: tuple[str]
    JOINT_TYPES: tuple[str]
    JOINT_SUFFIX: tuple[str]
    URDF_FILE: str
    LINK_LENGTHS: np.ndarray
    COM_OFFSET: np.ndarray
    HIP_OFFSETS: np.ndarray
    STANCE_HEIGHT: float
    STANCE_POSTURE: np.ndarray
    TORQUE_LIMITS: np.ndarray
    ROBOT_SIZE: np.ndarray

    def __init__(self, sim_env=pybullet, init_height_addition=0.0,
                 make_motor: make_cls = MotorSim):
        self._env, self._frequency = sim_env, g_cfg.execution_frequency
        self._motor: MotorSim = make_motor(self, num=12, frequency=self._frequency,
                                           kp=80, kd=(1.0, 2.0, 2.0) * 4)
        assert g_cfg.latency >= 0
        self._latency = g_cfg.latency

        self._resetStates()

        self._latency_steps = int(self._latency * g_cfg.execution_frequency)
        self._quadruped = self._loadRobot(init_height_addition)
        self._analyseModelJoints()
        self._resetPosture()
        self.setPhysicsParams()

        self._observation_history: Deque[ObservationRaw] = deque(maxlen=100)
        self._observation_noisy_history: Deque[ObservationRaw] = deque(maxlen=100)
        self._command_history: Deque[np.ndarray] = deque(maxlen=100)
        self._cot_buffer: Deque[float] = deque(maxlen=int(self._frequency * 0.1))

    @property
    def id(self):
        return self._quadruped

    def _resetStates(self):
        self._base_pose: Pose = None
        self._base_twist: Twist = None
        self._base_twist_in_base_frame: Twist = None
        self._rpy: Rpy = None
        self._last_torque: np.ndarray = np.zeros(12)
        self._disturbance: np.ndarray = np.zeros(3)
        self._last_stance_states: list[tuple[float, np.ndarray]] = [None] * 4
        self._strides = [(0., 0.)] * 4
        self._slips = [(0., 0.)] * 4
        self._observation: ObservationRaw = None
        self._time = 0.0
        self._step_counter = 0

    def _loadRobot(self, init_height_addition=0.0):
        if g_cfg.on_rack:
            pos = self.INIT_RACK_POSITION
        else:
            pos = self.INIT_POSITION.copy()
            pos[2] += init_height_addition
        flags = self._env.URDF_USE_SELF_COLLISION if g_cfg.self_collision_enabled else 0
        path = os.path.join(g_cfg.local_urdf, self.URDF_FILE)
        robot = self._env.loadURDF(path, pos, self.INIT_ORIENTATION, flags=flags)
        if g_cfg.on_rack:
            self._env.createConstraint(robot, -1, childBodyUniqueId=-1, childLinkIndex=-1,
                                       jointType=self._env.JOINT_FIXED,
                                       jointAxis=TP_ZERO3, parentFramePosition=TP_ZERO3,
                                       childFramePosition=self.INIT_RACK_POSITION,
                                       childFrameOrientation=self.INIT_ORIENTATION)
        return robot

    def _getJointIdByName(self, leg: str, joint_type: str):
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        if isinstance(joint_type, str):
            joint_type = self.JOINT_TYPES.index(joint_type)
        return self._getJointId(leg, joint_type)

    def _getJointId(self, leg: int | str, joint_type: int | str = 0):
        if joint_type < 0:
            joint_type += 4
        return self._joint_ids[leg * 4 + joint_type]

    def _analyseModelJoints(self):
        self._num_joints = self._env.getNumJoints(self._quadruped)
        self._joint_names = ['_'.join((l, j, s)) for l in self.LEG_NAMES
                             for j, s in zip(self.JOINT_TYPES, self.JOINT_SUFFIX)]

        joint_name_to_id = {}
        for i in range(self._env.getNumJoints(self._quadruped)):
            joint_info = self._env.getJointInfo(self._quadruped, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        self._joint_ids = [joint_name_to_id.get(n, -1) for n in self._joint_names]
        self._motor_ids = [self._getJointId(l, j) for l in range(4) for j in range(3)]
        self._foot_ids = [self._getJointId(l, -1) for l in range(4)]

    def _resetPosture(self):
        for i in range(12):
            self._env.resetJointState(self._quadruped, self._motor_ids[i], self.STANCE_POSTURE[i], 0.0)

    def setPhysicsParams(self, **kwargs):
        # for m in self._motor_ids:
        #     self._env.changeDynamics(self, -1, linearDamping=0, angularDamping=0)

        for f in self._foot_ids:
            self._env.changeDynamics(self._quadruped, f, spinningFriction=g_cfg.foot_spinning_friction,
                                     lateralFriction=g_cfg.foot_lateral_friction)
        self._env.setPhysicsEngineParameter(enableConeFriction=0)
        self._env.setJointMotorControlArray(self._quadruped, self._motor_ids, self._env.VELOCITY_CONTROL,
                                            forces=(g_cfg.joint_friction,) * len(self._motor_ids))
        for leg in range(4):
            self._env.enableJointForceTorqueSensor(self._quadruped, self._getJointId(leg, 3), True)

        self._mass = sum([self._env.getDynamicsInfo(self._quadruped, i)[0] for i in range(self._num_joints)])

    def is_safe(self) -> bool:  # FIXME: MOVE THIS TO TASK
        raise DeprecationWarning

    def applyCommand(self, motor_commands):
        # motor_commands = np.clip(motor_commands, self.STANCE_POSTURE - 0.3,
        #                          self.STANCE_POSTURE + 0.3)
        motor_commands = np.asarray(motor_commands)
        self._command_history.append(motor_commands)
        torques = self._motor.apply_command(motor_commands)
        self._env.setJointMotorControlArray(self._quadruped, self._motor_ids,
                                            self._env.TORQUE_CONTROL, forces=torques)
        self._last_torque = torques
        return torques

    def reset(self, height_addition=0.0, reload=False, in_situ=False):
        """
        clear state histories and restore the robot to the initial state in simulation.
        :param height_addition: The additional height of the robot at spawning, usually according to the terrain height.
        :param reload: Reload the urdf of the robot to simulation world if true.
        :param in_situ: Reset in situ if true.
        :return: Initial observation after reset.
        """
        self._resetStates()
        self._observation_history.clear()
        self._observation_noisy_history.clear()
        self._command_history.clear()
        self._cot_buffer.clear()
        if reload:
            self._quadruped = self._loadRobot(height_addition)
            self.setPhysicsParams()
        else:
            if g_cfg.on_rack:
                self._env.resetBasePositionAndOrientation(self._quadruped, self.INIT_RACK_POSITION, self.orientation)
                self._env.resetBaseVelocity(self._quadruped, TP_ZERO3, TP_ZERO3)
            elif in_situ:
                x, y, z = self.position[0], self.position[1], self.INIT_POSITION[2] + height_addition
                _, _, yaw = self._env.getEulerFromQuaternion(self.orientation)
                orn_q = self._env.getQuaternionFromEuler((0.0, 0.0, yaw))
                self._env.resetBasePositionAndOrientation(self._quadruped, (x, y, z), orn_q)
                self._env.resetBaseVelocity(self._quadruped, TP_ZERO3, TP_ZERO3)
            else:
                init_position = self.INIT_POSITION.copy()
                init_position[2] += height_addition
                self._env.resetBasePositionAndOrientation(self._quadruped, init_position, TP_Q0)
                self._env.resetBaseVelocity(self._quadruped, TP_ZERO3, TP_ZERO3)
        self._resetPosture()
        self._motor.reset()

    def updateObservation(self):
        self._step_counter += 1
        self._time += 1 / self._frequency
        self._base_pose = Pose(*self._env.getBasePositionAndOrientation(self._quadruped))
        self._base_twist = Twist(*self._env.getBaseVelocity(self._quadruped))
        self._base_twist_in_base_frame = Twist(self._rotateFromWorldToBase(self._base_twist.linear),
                                               self._rotateFromWorldToBase(self._base_twist.angular))
        self._rpy = Rpy.from_quaternion(self._base_pose.orientation)
        self._observation = ObservationRaw()
        self._observation.base_state = BaseState(self._base_pose, self._base_twist_in_base_frame)
        joint_states_raw = self._env.getJointStates(self._quadruped, range(self._num_joints))
        self._observation.joint_states = JointStates(*zip(*joint_states_raw))
        self._observation.foot_states = self._getFootStates()
        self._observation.contact_states = ContactStates(self._getContactStates())
        self._observation_history.append(self._observation)
        observation_noisy = self._estimateObservation()
        self._observation_noisy_history.append(observation_noisy)
        self._motor.update_observation()
        self._updateStridesAndSlipStates()
        return self._observation, observation_noisy

    def ik(self, leg, pos, frame='base'):
        """
        Calculate the inverse kinematic of certain leg by calling pybullet.calculateInverseKinematics.
        :param leg: The number of leg, ranging from [0, 4).
        :param pos: Desired end-effect position in specified frame.
        :param frame: The frame where position expressed in. Default 'base'; 'hip' and 'shoulder' supported.
        :return: Joint angles of corresponding leg.
        """
        pos = np.asarray(pos)
        if frame == 'base':
            pass
        elif frame == 'hip':
            pos += self.HIP_OFFSETS[leg]
        elif frame == 'shoulder':
            shoulder_length = self.LINK_LENGTHS[0]
            if self.LEG_NAMES[leg].endswith('R'):
                shoulder_length *= -1
            pos = pos + self.HIP_OFFSETS[leg] + (0, shoulder_length, 0)
        else:
            raise RuntimeError(f'Unknown Frame named {frame}')

        pos_world, _ = self._env.multiplyTransforms(self.position, self.orientation, pos, TP_Q0)
        all_joint_angles = self._env.calculateInverseKinematics(
            self._quadruped, self._foot_ids[leg], pos_world, solver=0)

        return all_joint_angles[leg * 3: leg * 3 + 3]

    def fk(self, leg, angles) -> Odometry:
        raise NotImplementedError

    def addDisturbanceOnBase(self, force, pos=TP_ZERO3):
        self._disturbance = force
        self._env.applyExternalForce(self._quadruped, -1, force, pos)

    def _rotateFromWorld(self, vector_world, reference):
        _, reference_inv = self._env.invertTransform(TP_ZERO3, reference)
        rotated, _ = self._env.multiplyTransforms(TP_ZERO3, reference_inv, vector_world, TP_Q0)
        return rotated

    def _rotateFromWorldToBase(self, vector_world):
        return self._rotateFromWorld(vector_world, self._base_pose.orientation)

    def _getContactStates(self):
        def _getContactState(link_id):
            return bool(self._env.getContactPoints(bodyA=self._quadruped, linkIndexA=link_id))

        contact_states = [_getContactState(range(self._num_joints))]
        return contact_states

    def _getFootStates(self):
        """
        Get foot positions, orientations and forces by getLinkStates and getContactPoints.
        :return: FootStates
        """
        link_states = self._env.getLinkStates(self._quadruped, self._foot_ids)
        foot_positions = [ls[0] for ls in link_states]
        foot_orientations = [ls[1] for ls in link_states]
        contact_points = self._env.getContactPoints(bodyA=self._quadruped)
        contact_dict = {}
        for p in contact_points:
            link_idx = p[3]
            directions = p[7], p[11], p[13]
            forces = p[9], p[10], p[12]
            contact_dict[link_idx] = (contact_dict.get(link_idx, (0., 0., 0.)) +
                                      sum(np.array(d) * f for d, f in zip(directions, forces)))
        foot_forces = [contact_dict.get(f, TP_ZERO3) for f in self._foot_ids]
        return FootStates(foot_positions, foot_orientations, foot_forces)

    def _updateStridesAndSlipStates(self):
        for i, c in enumerate(self.getFootContactStates()):
            if c:
                # if i == 0:
                #     sts = [self.getObservationHistoryFromIndex(-i).contact_states[3] for i in range(1, 10)]
                #     f = [np.linalg.norm(self.getObservationHistoryFromIndex(-i).foot_states.forces[0]) for i in
                #          range(1, 10)]
                #     print(np.array([sts, f]), sep='\n')
                foot_position = self.getFootPositionInWorldFrame(i)
                if self._last_stance_states[i]:
                    t, pos = self._last_stance_states[i]
                    if self._time - t >= 0.05:
                        self._strides[i] = (foot_position - pos)[:2]
                        self._slips[i] = (0., 0.)
                        # if i == 0:
                        #     print(self._time, t, self._strides[i], end='\n\n')
                    else:
                        self._slips[i] = (foot_position - pos)[:2]
                        self._strides[i] = (0., 0.)
                self._last_stance_states[i] = (self._time, foot_position)

    def _estimateObservation(self):
        # TODO: ADD NOISE
        idx = 0 if len(self._observation_history) <= self._latency_steps else -self._latency_steps - 1
        observation = self._observation_history[idx]
        return self._addNoiseOnObservation(observation)

    def _addNoiseOnObservation(self, observation):
        return observation

    @property
    def position(self):  # without latency and noise
        return self._base_pose.position

    @property
    def orientation(self):
        return self._base_pose.orientation

    @property
    def rpy(self):
        return self._rpy

    def getObservation(self, noisy=False):
        return self._observation_noisy_history[-1] if noisy else self._observation

    def getBasePosition(self, noisy=False):
        return self.getObservation(noisy).base_state.pose.position

    def getBaseOrientation(self, noisy=False):
        return self.getObservation(noisy).base_state.pose.orientation

    def getBaseAxisZ(self, noisy=False):
        return Rotation.from_quaternion(self.getBaseOrientation(noisy)).Z

    def getBaseRpy(self, noisy=False):
        return Rpy.from_quaternion(self.getBaseOrientation(noisy))

    def getHorizontalFrameInBaseFrame(self, noisy=False):
        rot = Rotation.from_quaternion(self.getBaseOrientation(noisy))
        y = self._rpy.y
        sy, cy = np.sin(y), np.cos(y)
        X = (cy, sy, 0)
        Y = (-sy, cy, 0)
        Z = (0, 0, 1)
        return rot.transpose() @ np.array((X, Y, Z)).transpose()

    def getBaseLinearVelocity(self):
        """
        Get the real robot linear velocity in world frame.
        :return: linear velocity in np.ndarray with shape (3,)
        """
        return self._base_twist.linear

    def getBaseAngularVelocity(self):
        return self._base_twist.angular

    def getBaseLinearVelocityInBaseFrame(self, noisy=False):
        return self.getObservation(noisy).base_state.twist.linear

    def getBaseAngularVelocityInBaseFrame(self, noisy=False):
        return self.getObservation(noisy).base_state.twist.angular

    def getBaseRpyRate(self):
        return get_rpy_rate_from_angular_velocity(self._rpy, self._base_twist.angular)

    def getBaseRpyRateInBaseFrame(self):
        return get_rpy_rate_from_angular_velocity(self._rpy, self._base_twist_in_base_frame.angular)

    def getContactStates(self):
        return self._observation.contact_states

    def getBaseContactState(self):
        return self._observation.contact_states[0]

    def getFootContactStates(self):
        return self._observation.contact_states[self._foot_ids,]

    def getPrevFootContactStates(self):
        return self.getObservationHistoryFromIndex(-2).contact_states[self._foot_ids,]

    def getFootContactForces(self):
        return self._observation.foot_states.forces.reshape(-1)

    def getFootPositionInBaseFrame(self, leg):
        joint_pos = self.getJointPositions(noisy=False)
        return self.fk(leg, joint_pos[leg * 3: leg * 3 + 3])

    def getFootPositionInWorldFrame(self, leg):
        return self._observation.foot_states.positions[leg, :]

    def getPrevFootPositionInWorldFrame(self, leg):
        return self.getObservationHistoryFromIndex(-2).foot_states.positions[leg, :]

    def getFootXYsInWorldFrame(self):
        return [self._observation.foot_states.positions[leg, :2] for leg in range(4)]

    def getFootSlipVelocity(self):  # NOTICE: THIS ACTUALLY INCLUDES FOOT ROLLING VELOCITIES
        return [np.linalg.norm(slip) * self._frequency for slip in self._slips]

    # def getFootSlipVelocity(self):
    #     current_contact = self.getFootContactStates()
    #     previous_contact = self.getPrevFootContactStates()
    #     calculate_slip = np.logical_and(current_contact, previous_contact)
    #     slip_velocity = np.zeros(4, dtype=float)
    #     for i, flag in enumerate(calculate_slip):
    #         if flag:
    #             current_leg_pos = self.getFootPositionInWorldFrame(i)
    #             previous_leg_pos = self.getPrevFootPositionInWorldFrame(i)
    #             current_leg_orn = self._observation.foot_states.orientations[i]
    #             previous_leg_orn = self.getObservationHistoryFromIndex(-2).foot_states.orientations[i]
    #             z, angle = pybullet.getAxisAngleFromQuaternion(
    #                 pybullet.getDifferenceQuaternion(previous_leg_orn, current_leg_orn))
    #             x = current_leg_pos - previous_leg_pos
    #             if np.linalg.norm(x) > 1e-5:
    #                 y = vec_cross(z, unit(x))
    #                 rotating_velocity = vec_cross(0.02 * y, z)
    #                 net_velocity = (current_leg_pos - previous_leg_pos) * self._frequency
    #                 slip_velocity[i] = np.linalg.norm((net_velocity - rotating_velocity)[:2])
    #             # np.set_printoptions(5)
    #             # print(current_leg_pos)
    #             # print((net_velocity - rotating_velocity)[:2])
    #             # print(net_velocity[:2])
    #             # print(i, np.linalg.norm(current_leg_position - previous_leg_position))
    #             # slip_velocity[i] = np.linalg.norm(current_leg_pos - previous_leg_pos) * self._frequency
    #             # print((current_leg_pos - previous_leg_pos) * self._frequency)
    #             # print()
    #     return slip_velocity

    def getStrides(self):
        return self._strides

    def getCostOfTransport(self):
        mgv = self._mass * 9.8 * np.linalg.norm(self._base_twist.linear)
        work = sum(filter(lambda i: i > 0, self._last_torque * self.getJointVelocities()))
        self._cot_buffer.append(1.0 if mgv == 0.0 else work / mgv)
        return np.mean(self._cot_buffer)

    def getJointStates(self) -> JointStates:
        return self._observation.joint_states

    def getJointPositions(self, noisy=False):
        return self.getObservation(noisy).joint_states.position[self._motor_ids,]

    def getJointVelocities(self, noisy=False):
        return self.getObservation(noisy).joint_states.velocity[self._motor_ids,]

    def getLastAppliedTorques(self):
        return self._last_torque

    def getBaseDisturbance(self):
        return self._disturbance

    def getCmdHistoryFromIndex(self, idx):
        len_requirement = -idx if idx < 0 else idx + 1
        return self.STANCE_POSTURE if len(self._command_history) < len_requirement else self._command_history[idx]

    def getObservationHistoryFromIndex(self, idx, noisy=False) -> ObservationRaw:
        len_requirement = -idx if idx < 0 else idx + 1
        idx = 0 if len(self._observation_history) < len_requirement else idx
        history = self._observation_noisy_history[idx] if noisy else self._observation_history[idx]
        return history

    def getJointPosHistoryFromIndex(self, idx, noisy=False):
        return self.getObservationHistoryFromIndex(idx, noisy).joint_states.position[self._motor_ids,]

    def getJointPosErrHistoryFromIndex(self, idx, noisy=False):
        return self.getCmdHistoryFromIndex(idx) - self.getJointPosHistoryFromIndex(idx, noisy)

    def getJointVelHistoryFromIndex(self, idx, noisy=False):
        return self.getObservationHistoryFromIndex(idx, noisy).joint_states.velocity[self._motor_ids,]

    def _getIndexFromMoment(self, moment):
        assert moment < 0
        return -1 - int((self._latency - moment) * self._frequency)

    def getCmdHistoryFromMoment(self, moment):
        return self.getCmdHistoryFromIndex(self._getIndexFromMoment(moment))

    def getObservationHistoryFromMoment(self, moment, noisy=False) -> ObservationRaw:
        return self.getObservationHistoryFromIndex(self._getIndexFromMoment(moment), noisy)

    def getJointPosHistoryFromMoment(self, moment, noisy=False):
        return self.getJointPosHistoryFromIndex(self._getIndexFromMoment(moment), noisy)

    def getJointPosErrHistoryFromMoment(self, moment, noisy=False):
        return self.getJointPosErrHistoryFromIndex(self._getIndexFromMoment(moment), noisy)

    def getJointVelHistoryFromMoment(self, moment, noisy=False):
        return self.getJointVelHistoryFromIndex(self._getIndexFromMoment(moment), noisy)

    def printJointInfos(self):
        for i in range(p.getNumJoints(self._quadruped)):
            print(JointInfo(p.getJointInfo(self._quadruped, i)))


class A1(Quadruped):
    INIT_POSITION = np.array((0, 0, .33))
    INIT_RACK_POSITION = np.array((0, 0, 1))
    INIT_ORIENTATION = np.array((0, 0, 0, 1))
    NUM_MOTORS = 12
    LEG_NAMES = ['FR', 'FL', 'RR', 'RL']
    JOINT_TYPES = ['hip', 'upper', 'lower', 'toe']
    JOINT_SUFFIX = ['joint', 'joint', 'joint', 'fixed']
    URDF_FILE = 'a1/a1.urdf'
    LINK_LENGTHS = np.array((0.08505, 0.2, 0.2))
    COM_OFFSET = -np.array((0.012731, 0.002186, 0.000515))
    HIP_OFFSETS = np.array(((0.183, -0.047, 0.), (0.183, 0.047, 0.),
                            (-0.183, -0.047, 0.), (-0.183, 0.047, 0.)))
    STANCE_HEIGHT = 0.3
    STANCE_POSTURE = np.array((0, 0.723, -1.445) * 4)
    TORQUE_LIMITS = np.array(((-33.5,) * 12, (33.5,) * 12))
    ROBOT_SIZE = np.array(((-0.3, 0.3), (-0.1, 0.1)))

    def ik_absolute(self, leg: int | str, pos, frame='base'):
        """
        Calculate the accurate inverse dynamics by the geometry. May fail if a solution doesn't exist.
        """
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        shoulder_length, thigh_length, shank_length = self.LINK_LENGTHS
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
            return hip_angle, normalize(angle_thigh), normalize(angle_shank)

        pos = np.asarray(pos)
        if frame == 'world':  # FIXME: COORDINATE TRANSFORMATION SEEMS TO BE WRONG
            return _ik_hip_frame(self._rotateFromWorldToBase(pos) - self.HIP_OFFSETS[leg])
        if frame == 'base':
            return _ik_hip_frame(pos - self.HIP_OFFSETS[leg])
        if frame == 'hip':
            return _ik_hip_frame(pos)
        if frame == 'shoulder':
            return _ik_hip_frame(pos + (0, shoulder_length, 0))
        raise RuntimeError(f'Unknown Frame named {frame}')

    def fk(self, leg: int, angles) -> Odometry:
        def _mdh_matrix(alpha, a, d, theta):
            ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
            return Odometry(((ct, -st, 0),
                             (st * ca, ct * ca, -sa),
                             (st * sa, ct * sa, ca)),
                            (a, -sa * d, ca * d))

        shoulder_length, thigh_length, shank_length = self.LINK_LENGTHS
        if self.LEG_NAMES[leg].endswith('L'):
            shoulder_length *= -1
        a1, a2, a3 = angles
        transformation = Odometry(((0, 0, 1),
                                   (-1, 0, 0),
                                   (0, -1, 0)),
                                  self.HIP_OFFSETS[leg]) @ \
                         _mdh_matrix(0, 0, 0, a1) @ \
                         _mdh_matrix(0, shoulder_length, 0, np.pi / 2) @ \
                         _mdh_matrix(-np.pi / 2, 0, 0, a2) @ \
                         _mdh_matrix(0, thigh_length, 0, a3) @ \
                         _mdh_matrix(0, shank_length, 0, 0) @ \
                         Odometry(((0, 0, -1),
                                   (-1, 0, 0),
                                   (0, 1, 0)))
        return transformation

    def _getContactStates(self):
        def _getContactState(link_id):
            return bool(self._env.getContactPoints(bodyA=self._quadruped, linkIndexA=link_id))

        base_contact = _getContactState(0)
        contact_states = [base_contact]
        # NOTICE: THIS IS SPECIFICALLY FOR A1
        for leg in range(4):  # FIXME: CONTACT JUDGEMENT SHOULD BE THOUGHT OVER
            base_contact = base_contact or _getContactState(leg * 5 + 1) or _getContactState(leg * 5 + 2)
            contact_states.extend([_getContactState(leg * 5 + i) for i in range(3, 6)])
        return contact_states

    def getFootContactStates(self):
        return self._observation.contact_states[(3, 6, 9, 12),]

    def getPrevFootContactStates(self):
        return self.getObservationHistoryFromIndex(-2).contact_states[(3, 6, 9, 12),]


if __name__ == '__main__':
    import time
    from burl.sim.terrain import RandomUniformTerrain

    g_cfg.on_rack = True
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    terrain = RandomUniformTerrain(p, size=30, downsample=5, resolution=0.05)
    make_motor = make_cls(MotorSim)
    q = A1(sim_env=p, make_motor=make_motor)
    # p.resetSimulation()
    # terrain.reset()
    # robot = p.loadURDF(A1.URDF_FILE, A1.INIT_POSITION, A1.INIT_ORIENTATION)
    # q.reset(reload=True)
    q.printJointInfos()
    p.setGravity(0, 0, -9.8)
    # c = p.loadURDF("cube.urdf", globalScaling=0.1)
    for _ in range(100000):
        p.stepSimulation()
        # q.updateObservation()
        time.sleep(1. / 240)
        # cmd0 = q.ik(0, (0, 0, -0.3), 'shoulder')
        # cmd1 = q.ik(1, (0, 0, -0.3), 'shoulder')
        # cmd2 = q.ik(2, (0, 0, -0.3), 'shoulder')
        # cmd3 = q.ik(3, (0, 0, -0.3), 'shoulder')
        # tq = q.applyCommand(np.concatenate([cmd0, cmd1, cmd2, cmd3]))
        # print(q.getHorizontalFrameInBaseFrame())
