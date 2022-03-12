from __future__ import annotations

import math
import os.path
import random
from collections import deque

import numpy as np
import pybullet
import pybullet_data

import burl
from burl.rl.state import JointStates, Pose, Twist, ContactStates, ObservationRaw, BaseState, FootStates
from burl.sim.motor import PdMotorSim, ActuatorNetSim, ActuatorNetWithHistorySim
from burl.utils import ang_norm, JointInfo, DynamicsInfo
from burl.utils.transforms import Rpy, Rotation, Odometry, get_rpy_rate_from_angular_velocity, Quaternion

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

    INIT_HEIGHT: float
    NUM_MOTORS = 12
    LEG_NAMES: tuple[str]
    JOINT_TYPES: tuple[str]
    JOINT_SUFFIX: tuple[str]
    URDF_FILE: str
    LINK_LENGTHS: tuple
    HIP_OFFSETS: tuple[tuple[float]]  # 4x3, hip joint coordinates in base frame
    STANCE_HEIGHT: float  # base height to foot joints when standing
    STANCE_FOOT_POSITIONS: tuple[tuple[float]]
    STANCE_POSTURE: tuple
    JOINT_LIMITS: tuple[tuple[float]]  # 2x3x4
    FOOT_RADIUS: float
    TORQUE_LIMITS: tuple
    ROBOT_SIZE: tuple
    P_PARAMS: float | tuple
    D_PARAMS: float | tuple

    WORLD_FRAME = -1
    BASE_FRAME = 0
    HIP_FRAME = 1
    SHOULDER_FRAME = 2
    INIT_FRAME = 3

    def __init__(self, sim_env=pybullet, execution_frequency=500,
                 latency=0., motor_latencies=(0., 0.),
                 random_dynamics=False, self_collision_enabled=False,
                 actuator_net=None):
        self._env, self._frequency = sim_env, execution_frequency
        motor_common_param = dict(frequency=execution_frequency,
                                  input_latency=motor_latencies[0], output_latency=motor_latencies[1],
                                  joint_limits=np.array(getattr(self, 'JOINT_LIMITS', None)).reshape(-1, 2),
                                  torque_limits=self.TORQUE_LIMITS)
        if not actuator_net:
            self._motor = PdMotorSim(self.P_PARAMS, self.D_PARAMS, **motor_common_param)
        elif actuator_net == 'single':
            self._motor = ActuatorNetSim(os.path.join(burl.rsc_path, 'actuator_net.pt'), **motor_common_param)
        elif actuator_net == 'history':
            self._motor = ActuatorNetWithHistorySim(os.path.join(burl.rsc_path, 'actuator_net_with_history.pt'),
                                                    **motor_common_param)
        else:
            raise NotImplementedError(f'Unknown Actuator Net Type {actuator_net}')
        self._latency = latency
        self._rand_dyn, self._self_collision = random_dynamics, self_collision_enabled
        self._resetStates()
        self._body_id = None

        self._observation_history: deque[ObservationRaw] = deque(maxlen=100)
        self._observation_noisy_history: deque[ObservationRaw] = deque(maxlen=100)
        self._command_history: deque[np.ndarray] = deque(maxlen=100)
        self._torque_history: deque[np.ndarray] = deque(maxlen=100)
        self._cot_buffer: deque[float] = deque(maxlen=int(2 * self._frequency))
        # self._cot_buffer: deque[float] = deque()

    def spawn(self, on_rack=False, position=(0., 0., 0.)):
        self._body_id = self._loadRobotOnRack() if on_rack else self._loadRobot(*position)
        self._analyseModelJoints()
        self._resetPosture()
        self.initPhysicsParams()
        self.randomDynamics() if self._rand_dyn else self.setDynamics()

    @property
    def id(self):
        return self._body_id

    def _resetStates(self):
        self._base_pose: Pose = None
        self._base_twist: Twist = None
        self._base_twist_Base: Twist = None
        self._rpy: Rpy = None
        self._last_stance_states: list[tuple[float, np.ndarray]] = [None] * 4
        self._max_foot_heights: np.ndarray = np.zeros(4)
        self._foot_clearances: np.ndarray = np.zeros(4)
        self._torque: np.ndarray = None
        self._strides = [(0., 0.)] * 4
        self._slips = [0.] * 4
        self._observation: ObservationRaw = None
        self._step_counter = -1
        self._sum_work = 0.0
        if isinstance(self._latency, float):
            self._latency_steps = int(self._latency * self._frequency)
        else:
            lower, upper = self._latency
            self._latency_steps = int(random.uniform(lower, upper) * self._frequency)

    def _loadRobotOnRack(self):
        path = os.path.join(burl.urdf_path, self.URDF_FILE)
        position, orientation = (0., 0., self.STANCE_HEIGHT * 2), (0., 0., 0., 1.)
        robot = self._env.loadURDF(path, position, orientation, flags=0)
        self._env.createConstraint(robot, -1, childBodyUniqueId=-1, childLinkIndex=-1,
                                   jointType=self._env.JOINT_FIXED,
                                   jointAxis=TP_ZERO3, parentFramePosition=TP_ZERO3,
                                   childFramePosition=position, childFrameOrientation=orientation)
        return robot

    def _loadRobot(self, x=0., y=0., altitude=0., orientation=TP_Q0):
        z = altitude + self.INIT_HEIGHT
        flags = pybullet.URDF_USE_SELF_COLLISION if self._self_collision else 0
        path = os.path.join(burl.urdf_path, self.URDF_FILE)
        return self._env.loadURDF(path, (x, y, z), orientation, flags=flags)

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

    def setDynamics(self, foot_lateral_friction=(0.4,) * 4,
                    foot_spinning_friction=(0.2,) * 4,
                    joint_friction=(0.025,) * 12):
        self._foot_friction = foot_lateral_friction
        for link_id, lf, sf in zip(self._foot_ids, self._foot_friction, foot_spinning_friction):
            self._env.changeDynamics(self._body_id, link_id, lateralFriction=lf, spinningFriction=sf)

        self._env.setJointMotorControlArray(self._body_id, self._motor_ids, self._env.VELOCITY_CONTROL,
                                            forces=joint_friction)
        for link_id in self._motor_ids:
            self._env.changeDynamics(self._body_id, link_id, linearDamping=0, angularDamping=0)
        self._mass = sum([self._env.getDynamicsInfo(self._body_id, i)[0] for i in range(self._num_joints)])

    def randomDynamics(self):
        base_mass = self._base_dynamics.mass * random.uniform(0.8, 1.2)
        base_inertia = self._base_dynamics.inertia * np.random.uniform(0.8, 1.2, 3)
        leg_masses, leg_inertia = zip(*[(leg_dyn.mass * random.uniform(0.8, 1.2),
                                         leg_dyn.inertia * np.random.uniform(0.8, 1.2, 3))
                                        for _ in range(4) for leg_dyn in self._leg_dynamics])

        self._env.changeDynamics(self._body_id, 0, mass=base_mass, localInertiaDiagonal=base_inertia)
        for link_id, mass, inertia in zip(self._motor_ids, leg_masses, leg_inertia):
            self._env.changeDynamics(self._body_id, link_id, mass=mass, localInertiaDiagonal=inertia)

        # self._latency = np.random.uniform(0.01, 0.02)
        self.setDynamics(foot_lateral_friction=np.random.uniform(0.4, 1.0, 4),
                         joint_friction=np.random.random(12) * 0.05)

    def applyCommand(self, motor_commands):
        motor_commands = np.asarray(motor_commands)
        self._command_history.append(motor_commands)
        torques = self._motor.apply_position(motor_commands)
        self.applyTorques(torques)
        return torques

    def applyTorques(self, torques):
        self._env.setJointMotorControlArray(self._body_id, self._motor_ids,
                                            self._env.TORQUE_CONTROL, forces=torques)
        self._torque = torques
        self._torque_history.append(torques)

    def reset(self, altitude=0.0, reload=False, in_situ=False):
        """
        clear state histories and restore the robot to the initial state in simulation.
        :param altitude: The additional height of the robot at spawning, usually according to the terrain height.
        :param reload: Reload the urdf of the robot to simulation world if true.
        :param in_situ: Reset in situ if true.
        :return: Initial observation after reset.
        """
        self._resetStates()
        self._observation_history.clear()
        self._observation_noisy_history.clear()
        self._command_history.clear()
        self._torque_history.clear()
        self._cot_buffer.clear()
        if reload:
            self._body_id = self._loadRobot(altitude)
            self.initPhysicsParams()
        else:
            z = self.INIT_HEIGHT + altitude
            if in_situ:
                x, y = self.position[:2]
                _, _, yaw = self._env.getEulerFromQuaternion(self.orientation)
                orn_q = self._env.getQuaternionFromEuler((0.0, 0.0, yaw))
                self._env.resetBasePositionAndOrientation(self._body_id, (x, y, z), orn_q)
            else:
                self._env.resetBasePositionAndOrientation(self._body_id, (0., 0., z), TP_Q0)
        if self._rand_dyn:
            self.randomDynamics()
        elif reload:
            self.setDynamics()
        self._env.resetBaseVelocity(self._body_id, TP_ZERO3, TP_ZERO3)
        self._resetPosture()
        self._motor.reset()

    def updateMinimalObservation(self):
        joint_states_raw = self._env.getJointStates(self._body_id, range(self._num_joints))
        joint_states = JointStates(*zip(*joint_states_raw))
        self._motor.update_observation(joint_states.position[self._motor_ids,],
                                       joint_states.velocity[self._motor_ids,])

    def updateObservation(self):
        self._step_counter += 1
        position, orientation = self._env.getBasePositionAndOrientation(self._body_id)
        self._rpy = Rpy.from_quaternion(orientation)
        self._base_pose = Pose(position, orientation, self._rpy)
        self._base_twist = Twist(*self._env.getBaseVelocity(self._body_id))
        self._base_twist_Base = Twist(self._rotateFromWorldToBase(self._base_twist.linear),
                                      self._rotateFromWorldToBase(self._base_twist.angular))
        self._observation = ObservationRaw()
        self._observation.base_state = BaseState(self._base_pose, self._base_twist, self._base_twist_Base)
        joint_states_raw = self._env.getJointStates(self._body_id, range(self._num_joints))
        self._observation.joint_states = JointStates(*zip(*joint_states_raw))
        self._observation.foot_states = self._getFootStates()
        self._observation.contact_states = ContactStates(self._getContactStates())
        self._observation_history.append(self._observation)
        observation_noisy = self._estimateObservation()
        self._observation_noisy_history.append(observation_noisy)
        self._motor.update_observation(self.getJointPositions(noisy=True),
                                       self.getJointVelocities(noisy=True))
        self._updateLocomotionInfos()
        return self._observation, observation_noisy

    def numericalInverseKinematics(self, leg, pos, frame=BASE_FRAME):
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

    def analyticalInverseKinematics(self, leg, pos, frame=BASE_FRAME):
        raise NotImplementedError

    def forwardKinematics(self, leg, angles) -> Odometry:
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

    def _updateLocomotionInfos(self):
        if self._torque is not None:
            mgv = self._mass * 9.8 * math.hypot(*self._base_twist.linear)
            work = sum(filter(lambda i: i > 0, self._torque * self.getJointVelocities()))
            self._sum_work += work
            self._cot_buffer.append(0.0 if mgv == 0.0 else work / mgv)
        joint_vel = self.getJointVelocities()
        rolling_vel = joint_vel[(1, 4, 7, 10),] + joint_vel[(2, 5, 8, 11),]
        for i, contact in enumerate(self.getFootContactStates()):
            foot_pos_world = self.getFootPositionInWorldFrame(i)
            if contact:
                if self._last_stance_states[i]:
                    step, pos = self._last_stance_states[i]
                    if (duration := (self._step_counter - step) / self._frequency) >= 0.05:
                        self._strides[i] = (foot_pos_world - pos)[:2]
                        self._slips[i] = 0.
                        self._foot_clearances[i] = self._max_foot_heights[i] - pos[2]
                    else:
                        # estimated slip distance
                        self._slips[i] = abs(math.hypot(*(foot_pos_world - pos)[:2]) -
                                             self.FOOT_RADIUS * rolling_vel[i] * duration)
                        self._strides[i] = (0., 0.)
                        self._foot_clearances[i] = 0.
                self._last_stance_states[i] = (self._step_counter, foot_pos_world)
            else:
                self._strides[i], self._slips[i], self._foot_clearances[i] = (0., 0.), 0., 0.
                self._max_foot_heights[i] = max(self._max_foot_heights[i], foot_pos_world[2])

    def _estimateObservation(self):
        idx = 0 if len(self._observation_history) <= self._latency_steps else -self._latency_steps - 1
        observation = self._observation_history[idx]
        # return observation
        observation_noisy = ObservationRaw(BaseState(), JointStates())
        add_noise = np.random.normal
        observation_noisy.base_state.pose = Pose()
        observation_noisy.base_state.pose.rpy = add_noise(observation.base_state.pose.rpy, 1e-2)
        observation_noisy.base_state.pose.orientation = Quaternion.from_rpy(observation_noisy.base_state.pose.rpy)
        observation_noisy.base_state.twist_Base = Twist(add_noise(observation.base_state.twist_Base.linear, 5e-2),
                                                        add_noise(observation.base_state.twist_Base.angular, 5e-2))
        observation_noisy.joint_states.position = add_noise(observation.joint_states.position, 5e-3)
        observation_noisy.joint_states.velocity = add_noise(observation.joint_states.velocity, 1e-1)
        return observation_noisy

    @property
    def position(self):  # without latency and noise
        return self._base_pose.position

    @property
    def orientation(self):
        return self._base_pose.orientation

    @property
    def rpy(self):
        return self._rpy

    def getFootFriction(self):
        return np.array(self._foot_friction)

    def getObservation(self, noisy=False):
        return self._observation_noisy_history[-1] if noisy else self._observation

    def getBasePosition(self):
        return self._base_pose.position

    def getBaseOrientation(self, noisy=False):
        return self.getObservation(noisy).base_state.pose.orientation

    def getGravityVector(self, noisy=False):
        return Rotation.from_quaternion(self.getBaseOrientation(noisy)).Z

    def getBaseRpy(self, noisy=False):
        return self.getObservation(noisy).base_state.pose.rpy if noisy else self._rpy

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
        return self.getObservation(noisy).base_state.twist_Base.linear

    def getBaseAngularVelocityInBaseFrame(self, noisy=False):
        return self.getObservation(noisy).base_state.twist_Base.angular

    def getBaseRpyRate(self):
        return get_rpy_rate_from_angular_velocity(self._rpy, self._base_twist.angular)

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
        return self.forwardKinematics(leg, joint_pos[leg * 3: leg * 3 + 3]).translation

    def getFootPositionInHipFrame(self, leg):
        return self.getFootPositionInBaseFrame(leg) - self.HIP_OFFSETS[leg]

    def getFootPositionInInitFrame(self, leg):
        shoulder_len = -self.LINK_LENGTHS[0] if self.LEG_NAMES[leg].endswith('R') else self.LINK_LENGTHS[0]
        return (self.getFootPositionInBaseFrame(leg) - self.HIP_OFFSETS[leg]
                - (0, shoulder_len, 0) - self.STANCE_FOOT_POSITIONS[leg])

    def getFootPositionInWorldFrame(self, leg):
        return self._observation.foot_states.positions[leg, :]

    def getPrevFootPositionInWorldFrame(self, leg):
        return self.getObservationHistoryFromIndex(-2).foot_states.positions[leg, :]

    def getFootXYsInWorldFrame(self):
        return [self._observation.foot_states.positions[leg, :2] for leg in range(4)]

    def getFootSlipVelocity(self):
        return np.array(self._slips) * self._frequency

    def getStrides(self):
        return np.array(self._strides)

    def getFootClearances(self):
        return self._foot_clearances

    def getCostOfTransport(self):
        return np.mean(self._cot_buffer).item() if self._cot_buffer else 0

    def getJointStates(self) -> JointStates:
        return self._observation.joint_states

    def getJointPositions(self, noisy=False):
        return self.getObservation(noisy).joint_states.position[self._motor_ids,]

    def getJointVelocities(self, noisy=False):
        return self.getObservation(noisy).joint_states.velocity[self._motor_ids,]

    def getJointAccelerations(self):
        if len(self._observation_history) > 2:
            return (self._observation.joint_states.velocity[self._motor_ids,] -
                    self._observation_history[-2].joint_states.velocity[self._motor_ids,]) * self._frequency
        return np.zeros(12)

    def getLastAppliedTorques(self):
        return self._torque

    def getTorqueGradients(self):
        if len(self._torque_history) > 2:
            return (self._torque - self._torque_history[-2]) * self._frequency
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
        for i in range(self._env.getNumJoints(self._body_id)):
            print(self._env.getDynamicsInfo(self._body_id, i))


class A1(Quadruped):
    INIT_HEIGHT = 0.33
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

    def analyticalInverseKinematics(self, leg: int | str, pos, frame=Quadruped.BASE_FRAME):
        """
        Calculate inverse kinematics analytically.
        """
        if isinstance(leg, str):
            leg = self.LEG_NAMES.index(leg)
        l_shoulder, l_thigh, l_shank = self.LINK_LENGTHS
        if self.LEG_NAMES[leg].endswith('R'):
            l_shoulder *= -1

        def _ik_hip_frame(_pos):
            while True:
                dx, dy, dz = _pos  # dz must lower than shoulder length
                l_stretch = math.sqrt((_pos ** 2).sum() - l_shoulder ** 2)
                a_hip_bias = math.atan2(dy, dz)
                try:
                    _sum = math.asin(l_shoulder / math.hypot(dy, dz))
                    a_hip = min(ang_norm(_sum - a_hip_bias), ang_norm(np.pi - _sum - a_hip_bias), key=abs)
                    a_stretch = -math.asin(dx / l_stretch)
                    a_shank = math.acos((l_shank ** 2 + l_thigh ** 2 - l_stretch ** 2) /
                                        (2 * l_shank * l_thigh)) - math.pi
                    a_thigh = a_stretch - math.asin(l_shank * math.sin(a_shank) / l_stretch)
                    break
                except ValueError:
                    _pos = _pos * 0.95
            # if hasattr(self, 'JOINT_LIMITS'):
            #     limits = self.JOINT_LIMITS[leg * 3: leg * 3 + 3]
            #     return np.clip((a_hip, a_thigh, a_shank), *zip(*limits))
            return np.array((a_hip, a_thigh, a_shank))

        pos = np.asarray(pos)
        if frame == Quadruped.WORLD_FRAME:  # FIXME: COORDINATE TRANSFORMATION SEEMS TO BE WRONG
            return _ik_hip_frame(self._rotateFromWorldToBase(pos) - self.HIP_OFFSETS[leg])
        if frame == Quadruped.BASE_FRAME:
            return _ik_hip_frame(pos - self.HIP_OFFSETS[leg])
        if frame == Quadruped.HIP_FRAME:
            return _ik_hip_frame(pos)
        if frame == Quadruped.SHOULDER_FRAME:
            return _ik_hip_frame(pos + (0, l_shoulder, 0))
        if frame == Quadruped.INIT_FRAME:
            return _ik_hip_frame(pos + (0, l_shoulder, 0) + self.STANCE_FOOT_POSITIONS[leg])
        raise RuntimeError(f'Unknown Frame {frame}')

    def forwardKinematics(self, leg: int, angles) -> Odometry:
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
    INIT_HEIGHT = 0.41
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
    JOINT_LIMITS = ((-1.22, 1.22), (-np.inf, np.inf), (-2.77, -0.7)) * 4
    TORQUE_LIMITS = ((-44.4,) * 12, (44.4,) * 12)
    ROBOT_SIZE = ((-0.325, 0.325), (-0.155, 0.155))
    P_PARAMS = 100
    D_PARAMS = 2

    # P_PARAMS = 150
    # D_PARAMS = 4

    def _getContactStates(self):
        def _getContactState(link_id):
            return bool(self._env.getContactPoints(bodyA=self._body_id, linkIndexA=link_id))

        base_contact = _getContactState(0)
        contact_states = [base_contact]
        for leg in range(4):  # FIXME: CONTACT JUDGEMENT SHOULD BE THOUGHT OVER
            base_contact = base_contact or _getContactState(leg * 4 + 2)
            contact_states.extend([_getContactState(leg * 4 + i) for i in range(3, 6)])
        return contact_states


if __name__ == '__main__':
    from burl.sim.terrain import PlainTerrain

    pybullet.connect(pybullet.GUI)
    pybullet.setTimeStep(2e-3)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    terrain = PlainTerrain(pybullet)
    robot = AlienGo()
    robot.spawn(on_rack=True)
    # robot.analyseJointInfos()
    pybullet.setGravity(0, 0, -9.8)

    for _ in range(100000):
        pybullet.stepSimulation()
        robot.updateObservation()
        # time.sleep(1. / 500)
        tq = robot.applyCommand(robot.STANCE_POSTURE)
