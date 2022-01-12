from __future__ import annotations

import math
import os.path
import random
from collections import deque
from typing import Deque

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from burl.rl.state import JointStates, Pose, Twist, ContactStates, ObservationRaw, BaseState, FootStates
from burl.sim.motor import MotorSim
from burl.utils import (ang_norm, JointInfo, make_cls, g_cfg, DynamicsInfo, vec_cross,
                        sign, included_angle, safe_asin, safe_acos)
from burl.utils.transforms import Rpy, Rotation, Odometry, get_rpy_rate_from_angular_velocity, Quaternion
from itertools import repeat

TP_ZERO3 = (0., 0., 0.)
TP_Q0 = (0., 0., 0., 1.)


class Quadruped(object):
    """
    A class for observing and controlling a quadruped in pybullet.

    For specific robot, the following attribute should be specified according to the urdf file.
    This class contains no pybullet.stepSimulation.
    Before a simulation cycle, run updateObservation() to set initial states.
    A general process is: applyCommand -> stepSimulation -> updateObservation.
    Some attribute like position represents some current real states; run methods like getJointHistory for history.
    Method updateObservation automatically saves real states and a noisy version with latency.
    Expect some commonly used attribute like position and orientation,
    it's suggested to use method with prefix 'get' to get observations.
    A 'get' method will give an accurate value, unless there's a 'noisy' option and 'noisy=True' is specified.
    """

    INIT_POSITION: tuple
    INIT_RACK_POSITION: tuple
    INIT_ORIENTATION: tuple
    NUM_MOTORS = 12
    LEG_NAMES: tuple[str]
    JOINT_TYPES: tuple[str]
    JOINT_SUFFIX: tuple[str]
    URDF_FILE: str
    LINK_LENGTHS: tuple
    HIP_OFFSETS: tuple[tuple[float]]  # 4 * 3, hip joint coordinates in base frame
    STANCE_HEIGHT: float  # base height to foot joints when standing
    STANCE_FOOT_POSITIONS: tuple[tuple[float]]
    STANCE_POSTURE: tuple
    FOOT_RADIUS: float
    TORQUE_LIMITS: tuple
    ROBOT_SIZE: tuple
    P_PARAMS: float | tuple
    D_PARAMS: float | tuple

    WORLD_FRAME = -1
    BASE_FRAME = 0
    HIP_FRAME = 1
    SHOULDER_FRAME = 2

    def __init__(self, sim_env=pybullet, init_height_addition=0.0,
                 make_motor: make_cls = MotorSim):
        self._env, self._frequency = sim_env, g_cfg.execution_frequency
        self._motor: MotorSim = make_motor(self, num=12, frequency=self._frequency,
                                           kp=self.P_PARAMS, kd=self.D_PARAMS, torque_limits=self.TORQUE_LIMITS)
        assert g_cfg.latency >= 0
        self._latency = g_cfg.latency

        self._resetStates()

        self._latency_steps = int(self._latency * g_cfg.execution_frequency)
        self._body_id = self._loadRobot(init_height_addition)
        self._analyseModelJoints()
        self._resetPosture()
        self.initPhysicsParams()
        self.setDynamics(g_cfg.random_dynamics)

        self._observation_history: Deque[ObservationRaw] = deque(maxlen=100)
        self._observation_noisy_history: Deque[ObservationRaw] = deque(maxlen=100)
        self._command_history: Deque[np.ndarray] = deque(maxlen=100)
        self._torque_history: Deque[np.ndarray] = deque(maxlen=100)
        # self._cot_buffer: Deque[float] = deque(maxlen=int(self._frequency / 1.25))
        self._cot_buffer: Deque[float] = deque()

    @property
    def id(self):
        return self._body_id

    def _resetStates(self):
        self._base_pose: Pose = None
        self._base_twist: Twist = None
        self._base_twist_in_base_frame: Twist = None
        self._rpy: Rpy = None
        self._last_stance_states: list[tuple[float, np.ndarray]] = [None] * 4
        self._max_foot_heights: np.ndarray = np.zeros(4)
        self._foot_clearances: np.ndarray = np.zeros(4)
        self._strides = [(0., 0.)] * 4
        self._slips = [(0., 0.)] * 4
        self._observation: ObservationRaw = None
        self._time = 0.0
        self._step_counter = 0
        self._sum_work = 0.0

    def _loadRobotOnRack(self):
        path = os.path.join(g_cfg.local_urdf, self.URDF_FILE)
        robot = self._env.loadURDF(path, self.INIT_RACK_POSITION, self.INIT_ORIENTATION, flags=0)
        self._env.createConstraint(robot, -1, childBodyUniqueId=-1, childLinkIndex=-1,
                                   jointType=self._env.JOINT_FIXED,
                                   jointAxis=TP_ZERO3, parentFramePosition=TP_ZERO3,
                                   childFramePosition=self.INIT_RACK_POSITION,
                                   childFrameOrientation=Quaternion.from_rpy((0., 0., 0.)))
        return robot

    def _loadRobot(self, init_height_addition=0.0):
        if g_cfg.on_rack:
            return self._loadRobotOnRack()
        x, y, z = self.INIT_POSITION
        z += init_height_addition
        flags = self._env.URDF_USE_SELF_COLLISION if g_cfg.self_collision_enabled else 0
        path = os.path.join(g_cfg.local_urdf, self.URDF_FILE)
        robot = self._env.loadURDF(path, (x, y, z), self.INIT_ORIENTATION, flags=flags)
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
        self._num_joints = self._env.getNumJoints(self._body_id)
        self._joint_names = ['_'.join((l, j, s)) for l in self.LEG_NAMES
                             for j, s in zip(self.JOINT_TYPES, self.JOINT_SUFFIX)]

        joint_name_to_id = {}
        for i in range(self._env.getNumJoints(self._body_id)):
            joint_info = self._env.getJointInfo(self._body_id, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        self._joint_ids = [joint_name_to_id.get(n, -1) for n in self._joint_names]
        self._motor_ids = [self._getJointId(l, j) for l in range(4) for j in range(3)]
        self._foot_ids = [self._getJointId(l, -1) for l in range(4)]

    def _resetPosture(self):
        for i in range(12):
            self._env.resetJointState(self._body_id, self._motor_ids[i], self.STANCE_POSTURE[i], 0.0)

    def initPhysicsParams(self):
        self._env.setPhysicsEngineParameter(enableConeFriction=0)
        for leg in range(4):
            self._env.enableJointForceTorqueSensor(self._body_id, self._getJointId(leg, 3), True)
        self._base_dynamics = DynamicsInfo(self._env.getDynamicsInfo(self._body_id, 0))
        self._leg_dynamics = [DynamicsInfo(self._env.getDynamicsInfo(self._body_id, i)) for i in self._motor_ids[:3]]

    def setDynamics(self, random_dynamics=False):
        if random_dynamics:
            base_mass = self._base_dynamics.mass * random.uniform(0.8, 1.2)
            base_inertia = self._base_dynamics.inertia * np.random.uniform(0.8, 1.2, 3)
            leg_masses, leg_inertia = zip(*[(leg_dyn.mass * random.uniform(0.8, 1.2),
                                             leg_dyn.inertia * np.random.uniform(0.8, 1.2, 3))
                                            for _ in range(4) for leg_dyn in self._leg_dynamics])
            joint_friction = np.random.random(12) * 0.05
            foot_friction = np.random.uniform(0.4, 1.0, 4)

            self._env.changeDynamics(self._body_id, 0, mass=base_mass, localInertiaDiagonal=base_inertia)
            # self._latency = np.random.uniform(0.01, 0.02)
            self._env.setJointMotorControlArray(self._body_id, self._motor_ids,
                                                pybullet.VELOCITY_CONTROL, forces=joint_friction)
            for link_id, mass, inertia in zip(self._motor_ids, leg_masses, leg_inertia):
                self._env.changeDynamics(self._body_id, link_id, mass=mass, localInertiaDiagonal=inertia)
            for link_id, fric in zip(self._foot_ids, foot_friction):
                self._env.changeDynamics(self._body_id, link_id, lateralFriction=fric)
        else:
            for link_id in self._foot_ids:
                self._env.changeDynamics(self._body_id, link_id, spinningFriction=g_cfg.foot_spinning_friction,
                                         lateralFriction=g_cfg.foot_lateral_friction)
            self._env.setJointMotorControlArray(self._body_id, self._motor_ids, self._env.VELOCITY_CONTROL,
                                                forces=(g_cfg.joint_friction,) * len(self._motor_ids))
        for link_id in self._motor_ids:
            self._env.changeDynamics(self._body_id, link_id, linearDamping=0, angularDamping=0)
        self._mass = sum([self._env.getDynamicsInfo(self._body_id, i)[0] for i in range(self._num_joints)])

    def applyCommand(self, motor_commands):
        motor_commands = np.asarray(motor_commands)
        self._command_history.append(motor_commands)
        torques = self._motor.apply_command(motor_commands)
        self._env.setJointMotorControlArray(self._body_id, self._motor_ids,
                                            self._env.TORQUE_CONTROL, forces=torques)
        self._torque_history.append(torques)
        return torques

    def reset(self, height_addition=0.0, reload=False, in_situ=False):
        """
        clear state histories and restore the robot to the initial state in simulation.
        :param height_addition: The additional height of the robot at spawning, usually according to the terrain height.
        :param reload: Reload the urdf of the robot to simulation world if true.
        :param in_situ: Reset in situ if true.
        :return: Initial observation after reset.
        """
        # print(np.mean(self._cot_buffer))
        self._resetStates()
        self._observation_history.clear()
        self._observation_noisy_history.clear()
        self._command_history.clear()
        self._torque_history.clear()
        self._cot_buffer.clear()
        if reload:
            self._body_id = self._loadRobot(height_addition)
            self.initPhysicsParams()
        else:
            if g_cfg.on_rack:
                self._env.resetBasePositionAndOrientation(self._body_id, self.INIT_RACK_POSITION, self.orientation)
            elif in_situ:
                x, y, z = self.position[0], self.position[1], self.INIT_POSITION[2] + height_addition
                _, _, yaw = self._env.getEulerFromQuaternion(self.orientation)
                orn_q = self._env.getQuaternionFromEuler((0.0, 0.0, yaw))
                self._env.resetBasePositionAndOrientation(self._body_id, (x, y, z), orn_q)
            else:
                x, y, z = self.INIT_POSITION
                self._env.resetBasePositionAndOrientation(self._body_id, (x, y, z + height_addition), TP_Q0)
        if reload or g_cfg.random_dynamics:
            self.setDynamics(g_cfg.random_dynamics)
        self._env.resetBaseVelocity(self._body_id, TP_ZERO3, TP_ZERO3)
        self._resetPosture()
        self._motor.reset()

    def updateObservation(self):
        self._step_counter += 1
        self._time += 1 / self._frequency
        self._base_pose = Pose(*self._env.getBasePositionAndOrientation(self._body_id))
        self._base_twist = Twist(*self._env.getBaseVelocity(self._body_id))
        self._base_twist_in_base_frame = Twist(self._rotateFromWorldToBase(self._base_twist.linear),
                                               self._rotateFromWorldToBase(self._base_twist.angular))
        self._rpy = Rpy.from_quaternion(self._base_pose.orientation)
        self._observation = ObservationRaw()
        self._observation.base_state = BaseState(self._base_pose, self._base_twist_in_base_frame)
        joint_states_raw = self._env.getJointStates(self._body_id, range(self._num_joints))
        self._observation.joint_states = JointStates(*zip(*joint_states_raw))
        self._observation.foot_states = self._getFootStates()
        self._observation.contact_states = ContactStates(self._getContactStates())
        self._observation_history.append(self._observation)
        observation_noisy = self._estimateObservation()
        self._observation_noisy_history.append(observation_noisy)
        self._motor.update_observation(self.getJointPositions(), self.getJointVelocities())
        self._updateStepInfos()
        return self._observation, observation_noisy

    def ik(self, leg, pos, frame=BASE_FRAME):
        """
        Calculate the inverse kinematic of certain leg by calling pybullet.calculateInverseKinematics.
        :param leg: The number of leg, ranging from [0, 4).
        :param pos: Desired end-effect position in specified frame.
        :param frame: The frame where position expressed in. Default BASE_FRAME; HIP_FRAME and SHOULDER_FRAME supported.
        :return: Joint angles of corresponding leg.
        """
        pos = np.asarray(pos)
        if frame == Quadruped.BASE_FRAME:
            pass
        elif frame == Quadruped.HIP_FRAME:
            pos += self.HIP_OFFSETS[leg]
        elif frame == Quadruped.SHOULDER_FRAME:
            shoulder_len = self.LINK_LENGTHS[0]
            if self.LEG_NAMES[leg].endswith('R'):
                shoulder_len *= -1
            pos = pos + self.HIP_OFFSETS[leg] + (0, shoulder_len, 0)
        else:
            raise RuntimeError(f'Unknown Frame {frame}')

        pos_world, _ = self._env.multiplyTransforms(self.position, self.orientation, pos, TP_Q0)
        all_joint_angles = self._env.calculateInverseKinematics(
            self._body_id, self._foot_ids[leg], pos_world, solver=0)

        return np.array(all_joint_angles[leg * 3: leg * 3 + 3])

    def ik_analytic(self, leg, pos, frame=BASE_FRAME):
        raise NotImplementedError

    def fk(self, leg, angles) -> Odometry:
        raise NotImplementedError

    def _rotateFromWorld(self, vector_world, reference):
        _, reference_inv = self._env.invertTransform(TP_ZERO3, reference)
        rotated, _ = self._env.multiplyTransforms(TP_ZERO3, reference_inv, vector_world, TP_Q0)
        return rotated

    def _rotateFromWorldToBase(self, vector_world):
        return self._rotateFromWorld(vector_world, self._base_pose.orientation)

    def _getContactStates(self):
        def _getContactState(link_id):
            return bool(self._env.getContactPoints(bodyA=self._body_id, linkIndexA=link_id))

        contact_states = [_getContactState(i) for i in range(self._num_joints)]
        return contact_states

    def _getFootStates(self):
        """
        Get foot positions, orientations and forces by getLinkStates and getContactPoints.
        :return: FootStates
        """
        link_states = self._env.getLinkStates(self._body_id, self._foot_ids)
        foot_positions = [ls[0] for ls in link_states]
        foot_orientations = [ls[1] for ls in link_states]
        contact_points = self._env.getContactPoints(bodyA=self._body_id)
        contact_dict = {}
        for p in contact_points:
            link_idx = p[3]
            directions = p[7], p[11], p[13]
            forces = p[9], p[10], p[12]
            contact_dict[link_idx] = (contact_dict.get(link_idx, (0., 0., 0.)) +
                                      sum(np.array(d) * f for d, f in zip(directions, forces)))
        foot_forces = [contact_dict.get(f, TP_ZERO3) for f in self._foot_ids]
        return FootStates(foot_positions, foot_orientations, foot_forces)

    def _updateStepInfos(self):
        for i, contact in enumerate(self.getFootContactStates()):
            foot_pos_world = self.getFootPositionInWorldFrame(i)
            if contact:
                if self._last_stance_states[i]:
                    t, pos = self._last_stance_states[i]
                    if self._time - t >= 0.05:
                        self._strides[i] = (foot_pos_world - pos)[:2]
                        self._slips[i] = (0., 0.)
                        self._foot_clearances[i] = self._max_foot_heights[i] - pos[2]
                    else:
                        self._slips[i] = (foot_pos_world - pos)[:2]
                        self._strides[i] = (0., 0.)
                        self._foot_clearances[i] = 0.
                self._last_stance_states[i] = (self._time, foot_pos_world)
            else:
                self._strides[i] = (0., 0.)
                self._foot_clearances[i] = 0.
                self._max_foot_heights[i] = max(self._max_foot_heights[i], foot_pos_world[2])

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
        return Rpy.from_quaternion(self.getBaseOrientation(True)) if noisy else self._rpy

    def transformFromHorizontalToBase(self, noisy=False):
        rot = Rotation.from_quaternion(self.getBaseOrientation(noisy))
        y = self._rpy.y
        sy, cy = np.sin(y), np.cos(y)
        X, Y, Z = (cy, sy, 0), (-sy, cy, 0), (0, 0, 1)
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
        return self.fk(leg, joint_pos[leg * 3: leg * 3 + 3]).translation

    def getFootPositionInHipFrame(self, leg):
        return self.getFootPositionInBaseFrame(leg) - self.HIP_OFFSETS[leg]

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
        return np.array(self._strides)

    def getFootClearances(self):
        return self._foot_clearances

    def getCostOfTransport(self):
        mgv = self._mass * 9.8 * math.hypot(*self._base_twist.linear)
        work = sum(filter(lambda i: i > 0, self._torque_history[-1] * self.getJointVelocities()))
        self._sum_work += work
        self._cot_buffer.append(0.0 if mgv == 0.0 else work / mgv)
        return np.mean(self._cot_buffer)

    def getJointStates(self) -> JointStates:
        return self._observation.joint_states

    def getJointPositions(self, noisy=False):
        return self.getObservation(noisy).joint_states.position[self._motor_ids,]

    def getJointVelocities(self, noisy=False):
        return self.getObservation(noisy).joint_states.velocity[self._motor_ids,]

    def getLastAppliedTorques(self):
        return self._torque_history[-1]

    def getTorqueGradients(self):
        if len(self._torque_history) > 2:
            return (self._torque_history[-1] - self._torque_history[-2]) * self._frequency
        return np.zeros(12)

    def getCmdHistoryFromIndex(self, idx):
        len_requirement = -idx if idx < 0 else idx + 1
        if len(self._command_history) < len_requirement:
            return np.array(self.STANCE_POSTURE)
        # print(self._command_history[-2], self._command_history[-1])
        return self._command_history[idx]

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

    def analyseJointInfos(self):
        print(f"{'id':>2}  {'name':^14} {'type':>4}  {'q u':^5}  {'damp'}  {'fric'}  {'range':^13} "
              f"{'maxF':>4}  {'maxV':>4}  {'link':^10}  {'AX'}  {'par'}  {'mass':>5}  {'inertial':^21}")
        for i in range(self._env.getNumJoints(self._body_id)):
            info = JointInfo(self._env.getJointInfo(self._body_id, i))
            is_fixed = info.type == pybullet.JOINT_FIXED
            print(f'{info.idx:>2d}  {info.name[:14]:^14} {info.joint_types[info.type]:>4}', end='  ')
            print(f"{f'{info.q_idx}':>2} {f'{info.u_idx}':<2} " if not is_fixed else f"{'---':^6}", end=' ')
            print(f"{f'{info.damping:.2f}':^4}  {f'{info.friction:.2f}':^4}  ", end='')
            print(f"{f'{info.limits[0]:.2f}':>5} ~{f'{info.limits[1]:.2f}':>5}  "
                  if not is_fixed else f"{'---':^13} ", end='')
            print(f"{f'{info.max_force:.1f}':>4}  {f'{info.max_vel:.1f}':>4}"
                  if not is_fixed else f"{'--':^4}  {f'--':^4}", end='  ')
            print(f"{f'{info.link_name[:10]}':^10}  ", end='')
            if not is_fixed:
                axis = info.axis_dict[info.axis] if info.axis in info.axis_dict else 'OT'
            else:
                axis = '--'
            print(f"{axis:>2}  {info.parent_idx:>2} ", end='  ')

            def float_e(f):
                e = 0
                while f < 1.0:
                    f *= 10
                    e += 1
                if e >= 10:
                    return f' {f:.0f}e-{e}'
                return f'{f:.1f}e-{e}'

            dyn = DynamicsInfo(self._env.getDynamicsInfo(self._body_id, i))
            print(f"{f'{dyn.mass:.3f}'}  "
                  f"{float_e(dyn.inertia[0])} {float_e(dyn.inertia[1])} {float_e(dyn.inertia[2])}")

    def analyseDynamicsInfos(self):
        for i in range(p.getNumJoints(self._body_id)):
            print(p.getDynamicsInfo(self._body_id, i))


class A1(Quadruped):
    INIT_POSITION = (0., 0., .33)
    INIT_RACK_POSITION = (0., 0., 1.)
    INIT_ORIENTATION = (0., 0., 0., 1.)
    NUM_MOTORS = 12
    LEG_NAMES = ('FR', 'FL', 'RR', 'RL')
    JOINT_TYPES = ('hip', 'upper', 'lower', 'toe')
    JOINT_SUFFIX = ('joint', 'joint', 'joint', 'fixed')
    URDF_FILE = 'a1/a1.urdf'
    LINK_LENGTHS = (0.08505, 0.2, 0.2)
    HIP_OFFSETS = ((0.183, -0.047, 0.), (0.183, 0.047, 0.),
                   (-0.183, -0.047, 0.), (-0.183, 0.047, 0.))
    STANCE_HEIGHT = 0.3
    FOOT_RADIUS = 0.02
    STANCE_FOOT_POSITIONS = ((0., 0., -STANCE_HEIGHT),) * 4
    STANCE_POSTURE = (0., 0.723, -1.445) * 4
    TORQUE_LIMITS = ((-33.5,) * 12, (33.5,) * 12)
    ROBOT_SIZE = ((-0.3, 0.3), (-0.1, 0.1))
    P_PARAMS = 80.
    D_PARAMS = (1., 2., 2.) * 4

    def ik_analytic(self, leg: int | str, pos, frame=Quadruped.BASE_FRAME):
        """
        Calculate inverse dynamics analytically.
        """
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        shoulder_len, thigh_len, shank_len = self.LINK_LENGTHS
        if self.LEG_NAMES[leg].endswith('R'):
            shoulder_len *= -1

        def _ik_hip_frame(_pos):
            dx, dy, dz = _pos
            distance = math.hypot(*_pos)
            hip_angle_bias = math.atan2(dy, dz)
            _sin = distance * shoulder_len / math.hypot(dy, dz) / distance
            _sum = safe_asin(_sin)
            hip_angle = min(ang_norm(_sum - hip_angle_bias), ang_norm(np.pi - _sum - hip_angle_bias), key=abs)
            shoulder_vector = np.array((0, math.cos(hip_angle), math.sin(hip_angle))) * shoulder_len
            foot_position_shoulder = _pos - shoulder_vector
            dist_foot_shoulder = math.hypot(*foot_position_shoulder)
            thigh_len_2, shank_len_2, dist_foot_shoulder_2 = \
                thigh_len ** 2, shank_len ** 2, dist_foot_shoulder ** 2
            _cos = (thigh_len_2 + shank_len_2 - dist_foot_shoulder_2) / (2 * thigh_len * shank_len)
            angle_shank = safe_acos(_cos) - np.pi

            _cos = (thigh_len_2 + dist_foot_shoulder_2 - shank_len_2) / (2 * thigh_len * dist_foot_shoulder)
            angle_thigh = safe_acos(_cos)
            normal = vec_cross(shoulder_vector, vec_cross((0, 0, -1), shoulder_vector))
            angle_thigh -= included_angle(normal, foot_position_shoulder) * sign(dx)
            return hip_angle, ang_norm(angle_thigh), ang_norm(angle_shank)

        pos = np.asarray(pos)
        if frame == Quadruped.WORLD_FRAME:  # FIXME: COORDINATE TRANSFORMATION SEEMS TO BE WRONG
            return _ik_hip_frame(self._rotateFromWorldToBase(pos) - self.HIP_OFFSETS[leg])
        if frame == Quadruped.BASE_FRAME:
            return _ik_hip_frame(pos - self.HIP_OFFSETS[leg])
        if frame == Quadruped.HIP_FRAME:
            return _ik_hip_frame(pos)
        if frame == Quadruped.SHOULDER_FRAME:
            return _ik_hip_frame(pos + (0, shoulder_len, 0))
        raise RuntimeError(f'Unknown Frame {frame}')

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
        transformation = (Odometry(((0, 0, 1),
                                    (-1, 0, 0),
                                    (0, -1, 0)),
                                   self.HIP_OFFSETS[leg]) @
                          _mdh_matrix(0, 0, 0, a1) @
                          _mdh_matrix(0, shoulder_length, 0, np.pi / 2) @
                          _mdh_matrix(-np.pi / 2, 0, 0, a2) @
                          _mdh_matrix(0, thigh_length, 0, a3) @
                          _mdh_matrix(0, shank_length, 0, 0) @
                          Odometry(((0, 0, -1),
                                    (-1, 0, 0),
                                    (0, 1, 0))))
        return transformation

    def _getContactStates(self):
        def _getContactState(link_id):
            return bool(self._env.getContactPoints(bodyA=self._body_id, linkIndexA=link_id))

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


class AlienGo(A1):
    INIT_POSITION = (0., 0., .41)
    INIT_RACK_POSITION = (0., 0., 1.)
    INIT_ORIENTATION = (0., 0., 0., 1.)
    NUM_MOTORS = 12
    LEG_NAMES = ('FR', 'FL', 'RR', 'RL')
    JOINT_TYPES = ('hip', 'thigh', 'calf', 'foot')
    JOINT_SUFFIX = ('joint', 'joint', 'joint', 'fixed')
    URDF_FILE = 'aliengo/urdf/aliengo.urdf'
    LINK_LENGTHS = (0.083, 0.25, 0.25)
    HIP_OFFSETS = ((0.2399, -0.051, 0.), (0.2399, 0.051, 0.),
                   (-0.2399, -0.051, 0), (-0.2399, 0.051, 0.))
    STANCE_HEIGHT = 0.4
    FOOT_RADIUS = 0.02649
    STANCE_FOOT_POSITIONS = ((0., 0., -STANCE_HEIGHT),) * 4
    STANCE_POSTURE = (0., 0.6435, -1.287) * 4
    TORQUE_LIMITS = ((-44.4,) * 12, (44.4,) * 12)
    ROBOT_SIZE = ((-0.325, 0.325), (-0.155, 0.155))
    P_PARAMS = 100.
    D_PARAMS = (2., 2., 2.) * 4

    # P_PARAMS = 160.
    # D_PARAMS = (2., 4., 4.) * 4
    # D_PARAMS = (2., 1., .5) * 4

    def _getContactStates(self):
        def _getContactState(link_id):
            return bool(self._env.getContactPoints(bodyA=self._body_id, linkIndexA=link_id))

        base_contact = _getContactState(0)
        contact_states = [base_contact]
        # NOTICE: THIS IS SPECIFICALLY FOR A1
        for leg in range(4):  # FIXME: CONTACT JUDGEMENT SHOULD BE THOUGHT OVER
            base_contact = base_contact or _getContactState(leg * 4 + 2)
            contact_states.extend([_getContactState(leg * 4 + i) for i in range(3, 6)])
        return contact_states


if __name__ == '__main__':
    import time
    from burl.sim.terrain import PlainTerrain

    g_cfg.on_rack = False
    g_cfg.action_frequency = 100.
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    terrain = PlainTerrain(p)
    make_motor = make_cls(MotorSim)
    q = AlienGo(sim_env=p, make_motor=make_motor)
    # p.resetSimulation()
    # terrain.reset()
    # robot = p.loadURDF(A1.URDF_FILE, A1.INIT_POSITION, A1.INIT_ORIENTATION)
    # q.reset(reload=True)
    q.analyseJointInfos()
    p.setGravity(0, 0, -9.8)
    # print(q.fk(0, q.STANCE_POSTURE[:3]).translation - q.HIP_OFFSETS[0] + (0., q.LINK_LENGTHS[0], 0.))
    # print(q.fk(1, q.STANCE_POSTURE[3:6]).translation - q.HIP_OFFSETS[1] + (0., -q.LINK_LENGTHS[0], 0.))
    # print(q.fk(2, q.STANCE_POSTURE[6:9]).translation - q.HIP_OFFSETS[2] + (0., q.LINK_LENGTHS[0], 0.))
    # print(q.fk(3, q.STANCE_POSTURE[9:12]).translation - q.HIP_OFFSETS[3] + (0., -q.LINK_LENGTHS[0], 0.))

    # c = p.loadURDF("cube.urdf", globalScaling=0.1)
    for _ in range(100000):
        p.stepSimulation()
        q.updateObservation()
        time.sleep(1. / 240)
        tq = q.applyCommand(q.STANCE_POSTURE)
        # print(q.position, q.rpy)
        # print(q.ik_absolute(0, (0, 0, -0.4), 'shoulder'))
        # print(q.getHorizontalFrameInBaseFrame())
