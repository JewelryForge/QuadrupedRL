from __future__ import annotations

import math
import time
from collections import deque
from itertools import chain

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from burl.rl.state import StateSnapshot, ProprioObservation, ExtendedObservation, Action
from burl.rl.task import BasicTask
from burl.sim.quadruped import A1, AlienGo, Quadruped
from burl.sim.terrain import makeTerrain
from burl.sim.tg import TgStateMachine, vertical_tg
from burl.utils import make_cls, g_cfg, log_info, log_debug, unit, vec_cross
from burl.utils.transforms import Rpy, Rotation


class QuadrupedEnv(object):
    """
    Manage a simulation environment of a Quadruped robot, including physics and rendering parameters.
    Reads g_cfg.trn_type to generate terrains.
    """

    ALLOWED_OBSERVATION_TYPES = {'snapshot': 'makeStateSnapshot',
                                 'proprio': 'makeProprioObservation',
                                 'noisy_proprio': 'makeNoisyProprioObservation',
                                 'extended': 'makeExtendedObservation',
                                 'noisy_extended': 'makeNoisyExtendedObservation'}

    def __init__(self, make_robot=Quadruped, make_task=BasicTask, observation_type=('snapshot',)):
        self._gui = g_cfg.rendering
        self._env = bullet_client.BulletClient(pybullet.GUI if self._gui else pybullet.DIRECT) if True else pybullet
        self._env.setAdditionalSearchPath(pybullet_data.getDataPath())
        for obs_type in observation_type:
            assert obs_type in self.ALLOWED_OBSERVATION_TYPES, f'Unknown Observation Type {obs_type}'
        self._observation_type = observation_type
        # self._loadEgl()
        if self._gui:
            self._prepareRendering()
        self._terrain = makeTerrain(self._env)
        self._robot: Quadruped = make_robot(self._env, execution_frequency=g_cfg.execution_frequency,
                                            on_rack=g_cfg.on_rack, random_dynamics=g_cfg.random_dynamics,
                                            motor_latencies=g_cfg.motor_latencies,
                                            height_addition=self._terrain.getPeakInRegion(*make_robot.ROBOT_SIZE)[2],
                                            actuator_net=g_cfg.actuator_net)
        self._task = make_task(self)
        assert g_cfg.sim_frequency >= g_cfg.execution_frequency >= g_cfg.action_frequency

        self._setPhysicsParameters()
        self._initSimulation()
        self._num_action_repeats = int(g_cfg.sim_frequency / g_cfg.action_frequency)
        self._num_execution_repeats = int(g_cfg.sim_frequency / g_cfg.execution_frequency)
        log_debug(f'Action Repeats for {self._num_action_repeats} time(s)')
        log_debug(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        self._resetStates()
        if self._gui:
            self._initRendering()
        self._action_buffer = deque(maxlen=10)
        self._external_force = np.array((0., 0., 0.))
        self._external_torque = np.array((0., 0., 0.))

    def _resetStates(self):
        self._sim_step_counter = 0
        self._episode_reward = 0.0
        self._is_failed = False
        self._est_X = None
        self._est_Y = None
        self._est_Z = None
        self._est_height = 0.0

    sim_time = property(lambda self: self._sim_step_counter / g_cfg.sim_frequency)
    sim_step = property(lambda self: self._sim_step_counter)
    client = property(lambda self: self._env)
    robot = property(lambda self: self._robot)
    terrain = property(lambda self: self._terrain)
    task = property(lambda self: self._task)
    is_failed = property(lambda self: self._is_failed)

    def initObservation(self):
        self._task.onInit()
        self._robot.updateObservation()
        return self.makeObservation()

    def _initSimulation(self):
        return
        for _ in range(300):
            self._robot.applyTorques((0.,) * 12)
            self._env.stepSimulation()

    def _loadEgl(self):
        import pkgutil
        if egl := pkgutil.get_loader('eglRenderer'):
            log_info(f'LoadPlugin: {egl.get_filename()}_eglRendererPlugin')
            self._env.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
        else:
            self._env.loadPlugin("eglRendererPlugin")

    def _prepareRendering(self):
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, False)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, False)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, False)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)

    def _initRendering(self):
        self._env.resetDebugVisualizerCamera(1.5, math.pi / 2, 0., (0., 0., self._robot.STANCE_HEIGHT))
        if g_cfg.extra_visualization:
            self._contact_visual_shape = self._env.createVisualShape(shapeType=pybullet.GEOM_BOX,
                                                                     halfExtents=(0.03, 0.03, 0.03),
                                                                     rgbaColor=(0.8, 0., 0., 0.6))
            self._terrain_visual_shape = self._env.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                                     radius=0.01,
                                                                     rgbaColor=(0., 0.8, 0., 0.6))
            self._contact_obj_ids = []
            self._terrain_indicators = [self._env.createMultiBody(baseVisualShapeIndex=self._terrain_visual_shape)
                                        for _ in range(36)]

        if g_cfg.show_time_ratio:
            self._last_time_ratio = 0.
            self._time_ratio_indicator = -1

        if g_cfg.show_indicators:
            self._force_indicator = -1
            self._torque_indicator = -1
            self._tip_axis_x = None  # torque indicator plane
            self._tip_axis_y = None
            self._tip_last_end = None
            self._tip_phase = 0
            self._cmd_indicators = [-1] * 7
            self._external_force_buffer = None
            self._external_torque_buffer = None
            self._cmd_buffer = None

        self._dbg_reset = self._env.addUserDebugParameter('reset', 1, 0, 0)
        self._reset_counter = 0

        self._last_frame_time = time.time()
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, True)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, True)

    def _setPhysicsParameters(self):
        # self._env.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
        self._env.setTimeStep(1 / g_cfg.sim_frequency)
        self._env.setGravity(0, 0, -9.8)

    def _updateRendering(self):
        if (current := self._env.readUserDebugParameter(self._dbg_reset)) != self._reset_counter:
            self._reset_counter = current
            self.reset()
        time_spent = time.time() - self._last_frame_time
        period_coeff = 1. if g_cfg.single_step_rendering else self._num_action_repeats
        if g_cfg.sleeping_enabled:
            period = period_coeff / g_cfg.sim_frequency / g_cfg.time_ratio
            if (time_to_sleep := period - time_spent) > 0:
                time.sleep(time_to_sleep)
                time_spent += time_to_sleep
        self._last_frame_time = time.time()
        time_ratio = period_coeff / g_cfg.sim_frequency / time_spent
        # print('time ratio:', f'{time_ratio:.2f}')
        if g_cfg.moving_camera:
            yaw, pitch, dist = self._env.getDebugVisualizerCamera()[8:11]
            (x, y, _), z = self._robot.position, self._robot.STANCE_HEIGHT
            self._env.resetDebugVisualizerCamera(dist, yaw, pitch, (x, y, z))

        if g_cfg.show_time_ratio:
            if self._last_time_ratio != time_ratio:
                _time_ratio_indicator = self._env.addUserDebugText(
                    f'{time_ratio: .2f}', textPosition=(0., 0., 0.), textColorRGB=(1., 1., 1.),
                    textSize=1, lifeTime=0, parentObjectUniqueId=self._robot.id,
                    replaceItemUniqueId=self._time_ratio_indicator)
                if self._time_ratio_indicator != -1 and _time_ratio_indicator != self._time_ratio_indicator:
                    self._env.removeUserDebugItem(self._time_ratio_indicator)
                self._time_ratio_indicator = _time_ratio_indicator
                self._last_time_ratio = time_ratio

        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, True)
        if g_cfg.extra_visualization:
            for obj in self._contact_obj_ids:
                self._env.removeBody(obj)
            self._contact_obj_ids.clear()
            for cp in self._env.getContactPoints(bodyA=self._robot.id):
                pos, normal, normal_force = cp[5], cp[7], cp[9]
                if normal_force > 0.1:
                    obj = self._env.createMultiBody(baseVisualShapeIndex=self._contact_visual_shape,
                                                    basePosition=pos)
                    self._contact_obj_ids.append(obj)
            positions = chain(*[self.getAbundantTerrainInfo(x, y, self._robot.rpy.y)
                                for x, y in self._robot.getFootXYsInWorldFrame()])
            for idc, pos in zip(self._terrain_indicators, positions):
                self._env.resetBasePositionAndOrientation(idc, posObj=pos, ornObj=(0, 0, 0, 1))
        if g_cfg.show_indicators:
            if self._external_force_buffer is not self._external_force:
                _force_indicator = self._env.addUserDebugLine(
                    lineFromXYZ=(0., 0., 0.), lineToXYZ=self._external_force / 50, lineColorRGB=(1., 0., 0.),
                    lineWidth=3, lifeTime=1,
                    parentObjectUniqueId=self._robot.id,
                    replaceItemUniqueId=self._force_indicator)
                if self._force_indicator != -1 and _force_indicator != self._force_indicator:
                    self._env.removeUserDebugItem(self._force_indicator)
                self._force_indicator = _force_indicator
                self._external_force_buffer = self._external_force

            if (self._external_torque != 0).all():
                magnitude = math.hypot(*self._external_torque)
                axis_z = self._external_torque / magnitude
                assistant = np.array((0., 0., 1.) if any(axis_z != (0., 0., 1.)) else (1., 0., 0.))
                self._tip_axis_x = unit(vec_cross(axis_z, assistant))
                self._tip_axis_y = vec_cross(axis_z, self._tip_axis_x)
                if self._tip_last_end is None:
                    self._tip_last_end = self._tip_axis_x * magnitude / 20
                self._tip_phase += math.pi / 36
                tip_end = (self._tip_axis_x * math.cos(self._tip_phase) +
                           self._tip_axis_y * math.sin(self._tip_phase)) * magnitude / 20
                _torque_indicator = self._env.addUserDebugLine(
                    lineFromXYZ=self._tip_last_end, lineToXYZ=tip_end, lineColorRGB=(0., 0., 1.),
                    lineWidth=5, lifeTime=0.1,
                    parentObjectUniqueId=self._robot.id,
                    replaceItemUniqueId=self._torque_indicator)
                self._tip_last_end = tip_end

            if self._cmd_buffer is not (cmd := self._task.cmd.copy()):
                axis_x = np.array((*cmd[:2], 0))
                axis_y = np.array((-axis_x[1], axis_x[0], 0))
                last_end = axis_x * 0.1
                phase, phase_inc = 0, cmd[2] / 3
                for i, cmd_ind in enumerate(self._cmd_indicators):
                    if i < 5:
                        phase += phase_inc
                    elif i == 5:
                        phase += math.pi / 12
                    else:
                        phase -= math.pi / 6
                    inc = ((axis_x * math.cos(phase) + axis_y * math.sin(phase)) / 15) * (1 if i < 5 else -1)
                    end = last_end + inc
                    _cmd_indicator = self._env.addUserDebugLine(
                        lineFromXYZ=last_end, lineToXYZ=end, lineColorRGB=(1., 1., 0.),
                        lineWidth=5, lifeTime=1, parentObjectUniqueId=self._robot.id,
                        replaceItemUniqueId=cmd_ind)
                    if cmd_ind != -1 and _cmd_indicator != cmd_ind:
                        self._env.removeUserDebugItem(cmd_ind)
                    self._cmd_indicators[i] = _cmd_indicator
                    if i < 5:
                        last_end = end

    def setDisturbance(self, force=(0.,) * 3, torque=(0.,) * 3):
        self._external_force = np.asarray(force)
        self._external_torque = np.asarray(torque)

    def makeObservation(self):
        if isinstance(self._observation_type, str):
            return getattr(self, self.ALLOWED_OBSERVATION_TYPES[self._observation_type])()
        else:
            obs_dict = {}.fromkeys(self.ALLOWED_OBSERVATION_TYPES, None)
            observations = []
            for obs_type in self._observation_type:
                if obs_dict[obs_type] is None:
                    func_name = self.ALLOWED_OBSERVATION_TYPES[obs_type]
                    obs = getattr(self, func_name)()
                    obs_dict[obs_type] = obs
                else:
                    obs = obs_dict[obs_type]
                observations.append(obs.standard())
            return observations

    def makeStateSnapshot(self, obs=None, noisy=False):
        if obs is None:
            obs = StateSnapshot()
        r = self._robot
        obs.command = self._task.cmd
        obs.gravity_vector = r.getGravityVector(noisy)
        obs.base_linear = r.getBaseLinearVelocityInBaseFrame(noisy)
        obs.base_angular = r.getBaseAngularVelocityInBaseFrame(noisy)
        obs.joint_pos = r.getJointPositions(noisy)
        obs.joint_vel = r.getJointVelocities(noisy)
        return obs

    def makeProprioObservation(self, obs=None, noisy=False):
        raise NotImplementedError

    def makeNoisyProprioObservation(self, obs=None):
        return self.makeProprioObservation(obs, noisy=True)

    def makeExtendedObservation(self, obs=None, noisy=False):
        raise NotImplementedError

    def makeNoisyExtendedObservation(self, obs=None):
        return self.makeExtendedObservation(obs, noisy=True)

    def _estimateTerrain(self):
        self._est_X, self._est_Y, self._est_Z = [], [], []
        for x, y in self._robot.getFootXYsInWorldFrame():
            self._est_X.append(x)
            self._est_Y.append(y)
            self._est_Z.append(self.getTerrainHeight(x, y))
        x, y, _ = self._robot.position
        self._est_X.append(x)
        self._est_Y.append(y)
        self._est_Z.append(self.getTerrainHeight(x, y))
        self._est_height = np.mean(self._est_Z)

    def step(self, action):
        # NOTICE: ADDING LATENCY ARBITRARILY FROM A DISTRIBUTION IS NOT REASONABLE
        # NOTICE: SHOULD CALCULATE TIME_SPENT IN REAL WORLD; HERE USE FIXED TIME INTERVAL
        rewards = []
        reward_details = {}
        prev_action = self._action_buffer[-1] if self._action_buffer else np.array(self._robot.STANCE_POSTURE)
        action = np.asarray(action)
        self._action_buffer.append(action)
        for i in range(self._num_action_repeats):
            # update_execution = self._sim_step_counter % self._num_execution_repeats == 0
            update_execution = True  # FIXME: task.calculateReward needs robot.updateObservation
            if update_execution:
                if g_cfg.use_action_interpolation:
                    weight = (i + 1) / self._num_action_repeats
                    current_action = action * weight + prev_action * (1 - weight)
                    torques = self._robot.applyCommand(current_action)
                else:
                    torques = self._robot.applyCommand(action)

            self._addRandomDisturbanceOnRobot()
            self._env.stepSimulation()
            self._sim_step_counter += 1
            self._estimateTerrain()
            if update_execution:
                self._robot.updateObservation()
            rewards.append(self._task.calculateReward())
            self._task.onSimulationStep()
            for n, r in self._task.getRewardDetails().items():
                reward_details[n] = reward_details.get(n, 0) + r
            if self._gui and g_cfg.single_step_rendering:
                self._env.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, True)
                self._updateRendering()
        if self._gui and not g_cfg.single_step_rendering:
            self._updateRendering()
        for n in reward_details:
            reward_details[n] /= self._num_action_repeats
        self._is_failed = self._task.isFailed()
        time_out = not self._is_failed and self._sim_step_counter >= g_cfg.max_sim_iterations
        self._episode_reward += (mean_reward := np.mean(rewards))
        info = {'time_out': time_out,
                'reward_details': reward_details,
                'episode_reward': self._episode_reward}
        # if hasattr(self._terrain, 'difficulty'):
        #     info['difficulty'] = self._terrain.difficulty
        if task_info := self._task.onStep():
            info['task_info'] = task_info
        # log_debug(f'Step time: {time.time() - start}')
        return (*self.makeObservation(),
                mean_reward,
                self._is_failed or time_out,
                info)

    def _addRandomDisturbanceOnRobot(self):
        self._applied_link_id = 0

        self._env.applyExternalForce(objectUniqueId=self._robot.id,
                                     linkIndex=self._applied_link_id,
                                     forceObj=self._external_force,
                                     posObj=(0.0, 0.0, 0.0),
                                     flags=pybullet.LINK_FRAME)
        self._env.applyExternalTorque(objectUniqueId=self._robot.id,
                                      linkIndex=self._applied_link_id,
                                      torqueObj=self._external_torque,
                                      flags=pybullet.LINK_FRAME)

    def reset(self):
        # completely_reset = self._task.curriculumUpdate(self._sim_step_counter)
        completely_reset = False
        # is_failed = self._is_failed
        self._task.reset()
        self._resetStates()
        # if not is_failed:
        #     if g_cfg.random_dynamics:
        #         self._robot.randomDynamics()
        #     return self.makeObservation()
        if completely_reset:
            self._env.resetSimulation()
            self._setPhysicsParameters()
            self._terrain.reset()
        self._robot.reset(self._terrain.getPeakInRegion(*self._robot.ROBOT_SIZE)[2],
                          reload=completely_reset)
        self._initSimulation()
        self._robot.updateObservation()
        return self.makeObservation()

    def close(self):
        self._env.disconnect()

    def getActionMutation(self):
        if len(self._action_buffer) < 3:
            return 0.0
        actions = [self._action_buffer[-i - 1] for i in range(3)]
        return np.linalg.norm(actions[0] - 2 * actions[1] + actions[2]) * g_cfg.action_frequency ** 2

    def getAbundantTerrainInfo(self, x, y, yaw):
        interval = 0.1
        dx, dy = interval * np.cos(yaw), interval * np.sin(yaw)
        points = ((dx - dy, dx + dy), (dx, dy), (dx + dy, -dx + dy),
                  (-dy, dx), (0, 0), (dy, -dx),
                  (-dx - dy, dx - dy), (-dx, -dy), (-dx + dy, -dx - dy))
        return [(xp := x + dx, yp := y + dy, self.getTerrainHeight(xp, yp)) for dx, dy in points]

    def getTerrainScan(self, x, y, yaw):
        return [p[2] for p in self.getAbundantTerrainInfo(x, y, yaw)]

    def getTerrainHeight(self, x, y) -> float:
        return self._terrain.getHeight(x, y)

    def getSafetyHeightOfRobot(self) -> float:
        return self._robot.position[2] - self._est_height

    def getSafetyFootHeightsOfRobot(self) -> np.ndarray:
        foot_pos = [self._robot.getFootPositionInWorldFrame(i) for i in range(4)]
        return np.array([z - self.getTerrainHeight(x, y) - 0.02 for x, y, z in foot_pos])

    def getSafetyRpyOfRobot(self) -> Rpy:
        X, Y, Z = np.array(self._est_X), np.array(self._est_Y), np.array(self._est_Z)
        # Use terrain points to fit a plane
        A = np.zeros((3, 3))
        A[0, :] = np.sum(X ** 2), X @ Y, np.sum(X)
        A[1, :] = A[0, 1], np.sum(Y ** 2), np.sum(Y)
        A[2, :] = A[0, 2], A[1, 2], len(X)
        b = np.array((X @ Z, Y @ Z, np.sum(Z)))
        a, b, _ = np.linalg.solve(A, b)
        trn_Z = unit((-a, -b, 1))
        rot_robot = Rotation.from_quaternion(self._robot.orientation)
        trn_Y = vec_cross(trn_Z, rot_robot.X)
        trn_X = vec_cross(trn_Y, trn_Z)
        # (trn_X, trn_Y, trn_Z) is the transpose of rotation matrix, so there's no need to transpose again
        return Rpy.from_rotation(np.array((trn_X, trn_Y, trn_Z)) @ rot_robot)

    def getTerrainNormal(self, x, y) -> np.ndarray:
        return self._terrain.getNormal(x, y)


class IkEnv(QuadrupedEnv):
    def __init__(self, make_robot=A1, make_task=BasicTask, observation_type=('extended', 'extended'),
                 ik_type='analytical', horizontal_frame=False):
        super().__init__(make_robot, make_task, observation_type)
        self._commands = None
        if ik_type == 'analytical':
            self._ik = self._robot.analyticalInverseKinematics
        elif ik_type == 'numerical':
            self._ik = self._robot.numericalInverseKinematics
        else:
            raise RuntimeError(f'Unknown IK Type {ik_type}')
        self._horizontal_frame = horizontal_frame

    def step(self, des_pos):
        if not self._horizontal_frame:
            self._commands = np.concatenate([self._ik(i, pos, Quadruped.INIT_FRAME)
                                             for i, pos in enumerate(des_pos)])
        else:
            h2b = self._robot.transformFromHorizontalToBase(True)
            offsets = ((0., -self._robot.LINK_LENGTHS[0], 0.),
                       (0., self._robot.LINK_LENGTHS[0], 0.),
                       (0., -self._robot.LINK_LENGTHS[0], 0.),
                       (0., self._robot.LINK_LENGTHS[0], 0.))
            des_pos = np.array([h2b @ (des_p + offset) for des_p, offset in zip(des_pos, offsets)])
            self._commands = np.concatenate([self._ik(i, pos, Quadruped.HIP_FRAME)
                                             for i, pos in enumerate(des_pos)])

        return super().step(self._commands)

    def getLastCommand(self) -> np.ndarray:
        return self._commands


class FixedTgEnv(IkEnv):
    tg_types = {'A1': make_cls(vertical_tg, h=0.08),
                'AlienGo': make_cls(vertical_tg, h=0.12)}

    def __init__(self, make_robot=A1, make_task=BasicTask, observation_type=('noisy_extended', 'extended'),
                 ik_type='analytical', horizontal_frame=False):
        super().__init__(make_robot, make_task, observation_type, ik_type, horizontal_frame)
        self._stm = TgStateMachine(1 / g_cfg.action_frequency,
                                   self.tg_types[self._robot.__class__.__name__])

    def makeProprioObservation(self, obs=None, noisy=False):
        if obs is None:
            obs = ProprioObservation()
        obs = self.makeStateSnapshot(obs, noisy=noisy)
        r = self._robot
        obs.joint_prev_pos_err = r.getJointPosErrHistoryFromIndex(-1, noisy)
        obs.ftg_frequencies = self._stm.frequency
        obs.ftg_phases = np.concatenate((np.sin(self._stm.phases), np.cos(self._stm.phases)))
        obs.joint_pos_err_his = np.concatenate((r.getJointPosErrHistoryFromMoment(-0.01, noisy),
                                                r.getJointPosErrHistoryFromMoment(-0.02, noisy)))
        obs.joint_vel_his = np.concatenate((r.getJointVelHistoryFromMoment(-0.01, noisy),
                                            r.getJointVelHistoryFromMoment(-0.02, noisy)))
        obs.joint_pos_target = r.getCmdHistoryFromIndex(-1)
        obs.joint_prev_pos_target = r.getCmdHistoryFromIndex(-self._num_action_repeats - 1)
        obs.base_frequency = (self._stm.base_frequency,)
        return obs

    def makeExtendedObservation(self, obs=None, noisy=False):
        if obs is None:
            obs = ExtendedObservation()
        obs: ExtendedObservation = self.makeProprioObservation(obs, noisy=noisy)
        r = self._robot
        foot_xy = r.getFootXYsInWorldFrame()
        obs.terrain_scan = np.concatenate([self.getTerrainScan(x, y, r.rpy.y) for x, y in foot_xy])
        obs.terrain_normal = np.concatenate([self.getTerrainNormal(x, y) for x, y in foot_xy])
        obs.contact_states = r.getContactStates()[1:]
        obs.foot_contact_forces = r.getFootContactForces()
        obs.foot_friction_coeffs = r.getFootFriction()
        obs.external_force = self._external_force
        obs.external_torque = self._external_torque
        return obs

    def step(self, action: Action | np.ndarray):
        if not isinstance(action, Action):
            action = Action.from_array(action)
        self._stm.update(action.leg_frequencies)
        priori = self._stm.get_priori_trajectory().reshape(4, 3)
        des_pos = action.foot_pos_residuals.reshape(4, 3) + priori

        if g_cfg.plot_trajectory:
            self.plotFootTrajectories(des_pos)
        return super().step(des_pos)

    def reset(self):
        self._stm.reset()
        return super().reset()

    def plotFootTrajectories(self, des_pos):
        from burl.utils import plotTrajectories
        if not hasattr(self, '_plotter'):
            self._plotter = plotTrajectories()
        for i, flag in enumerate(self._stm.cycles == 5):
            if flag:
                x, y, z = des_pos[i]
                self._plotter(i, (x, z), 'r')
                x, y, z = self._robot.getFootPositionInHipFrame(i)
                self._plotter(i, (x, z), 'b')

    def _initSimulation(self):  # for the stability of the beginning
        for _ in range(500):
            self._robot.updateMinimalObservation()
            self._robot.applyCommand(self._robot.STANCE_POSTURE)
            self._env.stepSimulation()


if __name__ == '__main__':
    from burl.utils import init_logger, set_logger_level

    g_cfg.on_rack = False
    g_cfg.trn_type = 'plain'
    g_cfg.add_disturbance = False
    g_cfg.moving_camera = False
    g_cfg.actuator_net = 'history'
    g_cfg.extra_visualization = False
    g_cfg.test_profile()
    # g_cfg.slow_down_rendering()
    # g_cfg.motor_latencies = (2e-3, 0.)
    init_logger()
    set_logger_level('DEBUG')
    np.set_printoptions(precision=3, linewidth=1000)
    tg = True
    if tg:
        env = FixedTgEnv(AlienGo)
        env.initObservation()
        for i in range(1, 100000):
            *_, reset, _ = env.step(Action())
            # if reset:
            #     env.reset()
    else:
        env = QuadrupedEnv(AlienGo)
        env.initObservation()
        for i in range(1000):
            env.step(env.robot.STANCE_POSTURE)

        # for i in range(100):
        #     env.step(env.robot.getJointPositions())
        # init_pos = env.robot.getJointPositions()
        # des_pos = np.array(env.robot.STANCE_POSTURE)
        # duration = 1
        # for i in range(1000):
        #     time_spent = i / g_cfg.action_frequency
        #     if time_spent > duration:
        #         env.step(env.robot.STANCE_POSTURE)
        #     else:
        #         process = time_spent / duration
        #         env.step(des_pos * process + init_pos * (1 - process))

        # for i in range(100000):
        #     env.step(env.robot.STANCE_POSTURE)
        #     print(env.robot.rpy)
