import math
from collections import deque
from typing import Any, Callable, Optional as Opt, Type, Union

import numpy as np
import pybullet as pyb
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from burl.rl.task import BasicTask
from burl.sim.quadruped import A1, AlienGo, Quadruped
from burl.sim.state import ProprioObservation, RealWorldObservation, ExtendedObservation, Action, ProprioInfo
from burl.sim.tg import TgStateMachine, vertical_tg
from burl.utils import make_part, g_cfg, log_info, log_debug, unit, vec_cross, ARRAY_LIKE
from burl.utils.transforms import Rpy, Rotation, Quaternion

__all__ = ['Quadruped', 'A1', 'AlienGo', 'QuadrupedEnv', 'IkEnv', 'FixedTgEnv', 'robot_auto_maker']


def robot_auto_maker(robot_class: Type[Quadruped] = AlienGo,
                     latency=None,
                     random_dynamics=None,
                     motor_latencies=None,
                     actuator_net=None):
    if not latency:
        latency = g_cfg.latency_range
    if not random_dynamics:
        random_dynamics = g_cfg.random_dynamics
    if not motor_latencies:
        motor_latencies = g_cfg.motor_latencies
    if not actuator_net:
        actuator_net = g_cfg.actuator_net
    return make_part(robot_class, latency=latency, random_dynamics=random_dynamics,
                     motor_latencies=motor_latencies, actuator_net=actuator_net)


class QuadrupedEnv(object):
    """
    Manage a simulation environment of a Quadruped robot, including physics and rendering parameters.
    Provides interface for reinforcement learning, including making observation and calculating rewards.
    """

    ALLOWED_OBS_TYPES = {'proprio': lambda self, obs=None: self.makeProprioObs(obs),
                         'noisy_proprio': lambda self, obs=None: self.makeProprioObs(obs, noisy=True),
                         'proprio_info': lambda self, obs=None: self.makeProprioInfo(obs),
                         'noisy_proprio_info': lambda self, obs=None: self.makeProprioInfo(obs, noisy=True),
                         'realworld': lambda self, obs=None: self.makeRealWorldObs(obs),
                         'noisy_realworld': lambda self, obs=None: self.makeRealWorldObs(obs, noisy=True),
                         'extended': lambda self, obs=None: self.makeExtendedObs(obs),
                         'noisy_extended': lambda self, obs=None: self.makeExtendedObs(obs, noisy=True)}

    def __init__(self, make_robot: Callable[..., Quadruped],
                 make_task: Callable[..., BasicTask] = BasicTask,
                 obs_types: Union[tuple[str, ...], str] = ()):
        self._gui = g_cfg.rendering
        self._env = BulletClient(pyb.GUI if self._gui else pyb.DIRECT) if True else pyb  # for pylint
        self._env.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.setObservationTypes(obs_types)
        # self._loadEgl()
        if self._gui:
            self._prepareRendering()
            self._init_rendering = False
        self._resetStates()

        self._robot: Quadruped = make_robot(execution_frequency=self.exec_freq)
        self._task = make_task(env=self)
        self._terrain = self._task.make_terrain(g_cfg.trn_type)
        self._robot.spawn(self._env, g_cfg.on_rack)
        self.moveRobotOnTerrain()

        self._setPhysicsParameters()
        assert g_cfg.sim_frequency >= g_cfg.execution_frequency >= g_cfg.action_frequency
        self._num_action_repeats = int(g_cfg.sim_frequency / g_cfg.action_frequency)
        self._num_execution_repeats = int(g_cfg.sim_frequency / g_cfg.execution_frequency)
        log_debug(f'Action Repeats for {self._num_action_repeats} time(s)')
        log_debug(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        self._action_history = deque(maxlen=10)
        self._external_force = np.array((0., 0., 0.))
        self._external_torque = np.array((0., 0., 0.))
        self._task.on_init()

    def setObservationTypes(self, obs_types: Union[str, tuple[str]], *other_obs_types: str):
        if isinstance(obs_types, str):
            self._obs_types = (obs_types,) + other_obs_types
        else:
            self._obs_types = obs_types
        for obs_type in self._obs_types:
            assert obs_type in self.ALLOWED_OBS_TYPES, f'Unknown Observation Type {obs_type}'

    def _resetStates(self):
        self._sim_step_counter = 0
        self._episode_reward = 0.0
        self._is_failed = False
        self._terrain_samples = []
        self._est_height = 0.0
        self._est_robot_height = deque(maxlen=2)
        self._make_obs_buffer = {}

    sim_time = property(lambda self: self._sim_step_counter / g_cfg.sim_frequency)
    sim_step = property(lambda self: self._sim_step_counter)
    client = property(lambda self: self._env)
    robot = property(lambda self: self._robot)
    terrain = property(lambda self: self._terrain)
    task = property(lambda self: self._task)
    is_failed = property(lambda self: self._is_failed)
    action_freq = property(lambda self: g_cfg.action_frequency)
    sim_freq = property(lambda self: g_cfg.sim_frequency)
    exec_freq = property(lambda self: g_cfg.execution_frequency)

    def initObservation(self):
        self._prepareSimulation()
        self.updateObservation()
        return self.makeObservation()

    def moveRobotOnTerrain(self):
        if g_cfg.on_rack:
            return
        self._estimateTerrain(self._robot.retrieveFootXYsInWorldFrame())
        init_height = self._est_height + self._robot.INIT_HEIGHT
        orn = Quaternion.from_rotation(self.getLocalTerrainRotation()).inverse()
        while True:
            self._env.resetBasePositionAndOrientation(self._robot.id, (0., 0., init_height), orn)
            self._env.performCollisionDetection()
            if not self._env.getContactPoints(self._robot.id, self._terrain.id):
                break
            init_height += 0.01

    def robotStuckInTerrain(self):
        try:
            foot_xyz = self._robot.getFootPositionsInWorldFrame()
        except AttributeError:
            foot_xyz = self._robot.retrieveFootPositionsInWorldFrame()
        for x, y, z in foot_xyz:
            if z < self.getTerrainHeight(x, y) - 0.03:
                print(z, self.getTerrainHeight(x, y))
                return True
        return False

        # if not self._sim_step_counter:
        #     self._env.performCollisionDetection()
        # for contact_point in self._env.getContactPoints(self._robot.id, self._terrain.id):
        #     if contact_point[8] < -0.01:
        #         return True
        # return False

    def _prepareSimulation(self):
        pass
        # for _ in range(300):
        #     self._robot.applyTorques((0.,) * 12)
        #     self._env.stepSimulation()

    def _loadEgl(self):
        import pkgutil
        if egl := pkgutil.get_loader('eglRenderer'):
            log_info(f'LoadPlugin: {egl.get_filename()}_eglRendererPlugin')
            self._env.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
        else:
            self._env.loadPlugin("eglRendererPlugin")

    def _prepareRendering(self):
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, False)
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, False)
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, False)
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, False)
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)

    def _updateRendering(self):
        if not self._init_rendering:
            self._dbg_reset = self._env.addUserDebugParameter('reset', 1, 0, 0)
            self._reset_counter = 0

            self._env.resetDebugVisualizerCamera(
                1.5, 0., 0., (0., 0., self._robot.STANCE_HEIGHT + self.getTerrainHeight(0., 0.)))
            self._env.configureDebugVisualizer(pyb.COV_ENABLE_GUI, True)
            self._env.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, True)
            self._init_rendering = True
        if (current := self._env.readUserDebugParameter(self._dbg_reset)) != self._reset_counter:
            self._reset_counter = current
            self.reset()
        self._env.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True)

    def _setPhysicsParameters(self):
        # self._env.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
        self._env.setTimeStep(1 / self.sim_freq)
        self._env.setGravity(0, 0, -9.8)

    def setDisturbance(self, force=(0.,) * 3, torque=(0.,) * 3):
        self._external_force = np.asarray(force)
        self._external_torque = np.asarray(torque)

    def getDisturbance(self):
        return self._external_force, self._external_torque

    def makeObservation(self):
        observations = []
        for obs_type in self._obs_types:
            if obs_type not in self._make_obs_buffer:
                obs = self.ALLOWED_OBS_TYPES[obs_type](self)
                self._make_obs_buffer[obs_type] = obs
            else:
                obs = self._make_obs_buffer[obs_type]
            observations.append(obs.standard())
        return tuple(observations)

    def _setObsBaseAttr(self, obs, base_obs_type: str):
        if base_obs_type in self._make_obs_buffer:
            base_obs = self._make_obs_buffer[base_obs_type]
        else:
            base_obs = self.ALLOWED_OBS_TYPES[base_obs_type](self)
            self._make_obs_buffer[base_obs_type] = base_obs
        for k, v in base_obs.__dict__.items():
            if not k.startswith('_'):
                setattr(obs, k, v)
        return obs

    def makeProprioObs(self, obs=None, noisy=False):
        if obs is None:
            obs = ProprioObservation()
        r = self._robot
        obs.command = self._task.cmd
        obs.gravity_vector = r.getGravityVector(noisy)
        obs.base_linear = r.getBaseLinearVelocityInBaseFrame(noisy)
        obs.base_angular = r.getBaseAngularVelocityInBaseFrame(noisy)
        obs.joint_pos = r.getJointPositions(noisy)
        obs.joint_vel = r.getJointVelocities(noisy)
        return obs

    def makeNoisyProprioObs(self, obs=None):
        return self.makeProprioObs(obs, noisy=True)

    def makeProprioInfo(self, obs=None, noisy=False):
        raise NotImplementedError

    def makeNoisyProprioInfo(self, obs=None):
        return self.makeProprioInfo(obs, noisy=True)

    def makeRealWorldObs(self, obs=None, noisy=False):
        raise NotImplementedError

    def makeNoisyRealWorldObs(self, obs=None):
        return self.makeRealWorldObs(obs, noisy=True)

    def makeExtendedObs(self, obs=None, noisy=False):
        raise NotImplementedError

    def makeNoisyExtendedObs(self, obs=None):
        return self.makeExtendedObs(obs, noisy=True)

    def _estimateTerrain(self, xy_points=None):
        self._terrain_samples.clear()
        est_h = 0.
        if not xy_points:
            xy_points = self._robot.getFootXYsInWorldFrame()
        for x, y in xy_points:
            z = self.getTerrainHeight(x, y)
            est_h += z
            self._terrain_samples.append((x, y, z))
        self._est_height = est_h / len(xy_points)

    def updateObservation(self):
        self._make_obs_buffer.clear()
        self._robot.updateObservation()
        self._estimateTerrain()
        self._est_robot_height.append(self.getTerrainBasedHeightOfRobot())

    def step(self, action: ARRAY_LIKE) -> tuple[tuple[Any], float, bool, dict]:
        """
        Given motor angles, calculate motor torques, step simulation, optionally update rendering
        and get observations and rewards.
        :param action: desired motor angles.
        :return: a tuple, containing (*observations, reward, is_finished, info)
        """
        # NOTICE: ADDING LATENCY ARBITRARILY FROM A DISTRIBUTION IS NOT REASONABLE
        # NOTICE: SHOULD CALCULATE TIME_SPENT IN REAL WORLD; HERE USE FIXED TIME INTERVAL
        rewards = []
        reward_details = {}
        prev_action = self._action_history[-1] if self._action_history else np.array(self._robot.STANCE_POSTURE)
        action = np.asarray(action)
        self._action_history.append(action)
        for i in range(self._num_action_repeats):
            update_execution = self._sim_step_counter % self._num_execution_repeats == 0
            if update_execution:
                if g_cfg.use_action_interpolation:
                    weight = (i + 1) / self._num_action_repeats
                    current_action = action * weight + prev_action * (1 - weight)
                    torques = self._robot.applyCommand(current_action)
                else:
                    torques = self._robot.applyCommand(action)

            self._applyDisturbanceOnRobot()
            self._env.stepSimulation()
            self._sim_step_counter += 1
            if update_execution:
                self.updateObservation()
                rewards.append(self._task.calc_reward())
                for n, r in self._task.reward_details.items():
                    reward_details[n] = reward_details.get(n, 0) + r
            self._task.on_sim_step()
        if self._gui:
            self._updateRendering()
        for n in reward_details:
            reward_details[n] /= self._num_action_repeats
        self._is_failed = self._task.is_failed()
        time_out = not self._is_failed and self._sim_step_counter >= g_cfg.max_sim_iterations
        mean_reward = np.mean(rewards).item()
        self._episode_reward += mean_reward
        info = {'time_out': time_out,
                'reward_details': reward_details,
                'episode_reward': self._episode_reward}
        if task_info := self._task.on_step():
            info['task_info'] = task_info
        # log_debug(f'Step time: {time.time() - start}')
        return (self.makeObservation(),
                mean_reward,
                self._is_failed or time_out,
                info)

    def _applyDisturbanceOnRobot(self):
        self._applied_link_id = 0
        self._env.applyExternalForce(objectUniqueId=self._robot.id, linkIndex=self._applied_link_id,
                                     forceObj=self._external_force, posObj=(0.0, 0.0, 0.0), flags=pyb.LINK_FRAME)
        self._env.applyExternalTorque(objectUniqueId=self._robot.id, linkIndex=self._applied_link_id,
                                      torqueObj=self._external_torque, flags=pyb.LINK_FRAME)

    def reset(self):
        self._task.reset()
        self._resetStates()
        # if not is_failed:
        #     if g_cfg.random_dynamics:
        #         self._robot.randomDynamics()
        #     return self.makeObservation()
        self._robot.reset()
        self.moveRobotOnTerrain()
        self._prepareSimulation()
        self.updateObservation()
        return self.makeObservation()

    # def reload(self):
    #     self._task.reset()
    #     self._resetStates()
    #     self._env.resetSimulation()
    #     self._setPhysicsParameters()
    #     self._terrain.spawn(self._env)
    #     self._robot.reset(reload=True)
    #     self.moveRobotOnTerrain()
    #     self._prepareSimulation()
    #     self.updateObservation()
    #     return self.makeObservation()

    def close(self):
        self._env.disconnect()

    def getActionMutation(self):
        if len(self._action_history) < 3:
            return 0.0
        actions = [self._action_history[-i - 1] for i in range(3)]
        return np.linalg.norm(actions[0] - 2 * actions[1] + actions[2]) * self.action_freq ** 2

    def getAbundantTerrainInfo(self, x, y, yaw) -> list[tuple[float, float, float]]:
        interval = 0.1
        dx, dy = interval * math.cos(yaw), interval * math.sin(yaw)
        points = ((dx - dy, dx + dy), (dx, dy), (dx + dy, -dx + dy),
                  (-dy, dx), (0, 0), (dy, -dx),
                  (-dx - dy, dx - dy), (-dx, -dy), (-dx + dy, -dx - dy))
        return [(xp := x + dx, yp := y + dy, self.getTerrainHeight(xp, yp)) for dx, dy in points]

    def getTerrainScan(self, x, y, yaw):
        return [p[2] for p in self.getAbundantTerrainInfo(x, y, yaw)]

    def getTerrainHeight(self, x, y) -> float:
        return self._terrain.get_height(x, y)

    def getTerrainBasedVerticalVelocityOfRobot(self):
        return (self._est_robot_height[-1] - self._est_robot_height[-2]) * self.exec_freq

    def getTerrainBasedHeightOfRobot(self) -> float:
        return self._robot.position[2] - self._est_height

    def estimateLocalTerrainNormal(self) -> np.ndarray:
        X, Y, Z = np.array(self._terrain_samples).T
        A = np.zeros((3, 3))
        A[0, :] = np.sum(X ** 2), X @ Y, np.sum(X)
        A[1, :] = A[0, 1], np.sum(Y ** 2), np.sum(Y)
        A[2, :] = A[0, 2], A[1, 2], len(X)
        b = np.array((X @ Z, Y @ Z, np.sum(Z)))
        a, b, _ = np.linalg.solve(A, b)
        return unit((-a, -b, 1))

    def getLocalTerrainRotation(self) -> np.ndarray:
        trn_Z = self.estimateLocalTerrainNormal()
        trn_Y = vec_cross(trn_Z, (1., 0., 0.))
        trn_X = vec_cross(trn_Y, trn_Z)
        # (trn_X, trn_Y, trn_Z) is the transpose of rotation matrix, so there's no need to transpose again
        return np.array((trn_X, trn_Y, trn_Z))

    def getTerrainBasedRpyOfRobot(self) -> Rpy:
        trn_Z = self.estimateLocalTerrainNormal()
        rot_robot = Rotation.from_quaternion(self._robot.orientation)
        trn_Y = vec_cross(trn_Z, rot_robot.X)
        trn_X = vec_cross(trn_Y, trn_Z)
        return Rpy.from_rotation(np.array((trn_X, trn_Y, trn_Z)) @ rot_robot)

    def getTerrainNormal(self, x, y) -> np.ndarray:
        return self._terrain.get_normal(x, y)


class IkEnv(QuadrupedEnv):
    def __init__(self, make_robot: Callable[..., Quadruped], make_task=BasicTask,
                 obs_types=(), ik_type='analytical', horizontal_frame=False):
        super().__init__(make_robot, make_task, obs_types)
        self._commands: Opt[np.ndarray] = None
        if ik_type == 'analytical':
            self._ik = self._robot.analyticalInverseKinematics
        elif ik_type == 'numerical':
            self._ik = self._robot.numericalInverseKinematics
        else:
            raise RuntimeError(f'Unknown IK Type {ik_type}')
        if horizontal_frame:
            self.calculate_commands = self.calculateCommandsInHorizontalFrame
        else:
            self.calculate_commands = self.calculateCommands

    def calculateCommands(self, des_pos: ARRAY_LIKE) -> np.ndarray:
        return np.concatenate([self._ik(i, pos, Quadruped.INIT_FRAME) for i, pos in enumerate(des_pos)])

    def calculateCommandsInHorizontalFrame(self, des_pos: ARRAY_LIKE) -> np.ndarray:
        h2b = self._robot.transformFromHorizontalToBase(True)
        offsets = ((0., -self._robot.LINK_LENGTHS[0], 0.),
                   (0., self._robot.LINK_LENGTHS[0], 0.),
                   (0., -self._robot.LINK_LENGTHS[0], 0.),
                   (0., self._robot.LINK_LENGTHS[0], 0.))
        des_pos = np.array([h2b @ (des_p + offset) for des_p, offset in zip(des_pos, offsets)])
        return np.concatenate([self._ik(i, pos, Quadruped.HIP_FRAME) for i, pos in enumerate(des_pos)])

    def step(self, des_pos: ARRAY_LIKE):
        self._commands = self.calculate_commands(des_pos)
        return super().step(self._commands)

    def getLastCommand(self) -> np.ndarray:
        return self._commands


class FixedTgEnv(IkEnv):
    tg_types = {'A1': make_part(vertical_tg, h=0.08),
                'AlienGo': make_part(vertical_tg, h=0.12)}

    def __init__(self, make_robot: Callable[..., Quadruped],
                 make_task=BasicTask, obs_types=('noisy_extended', 'extended'),
                 ik_type='analytical', horizontal_frame=False):
        super().__init__(make_robot, make_task, obs_types, ik_type, horizontal_frame)
        self._stm = TgStateMachine(1 / self.action_freq,
                                   self.tg_types[self._robot.__class__.__name__])

    def makeProprioInfo(self, obs=None, noisy=False):
        if obs is None:
            obs = ProprioInfo()
        obs = self._setObsBaseAttr(obs, 'noisy_proprio' if noisy else 'proprio')
        obs.joint_pos_target = self._action_history[-1] if self._action_history else self._robot.STANCE_POSTURE
        obs.ftg_frequencies = self._stm.frequency
        obs.ftg_phases = np.concatenate((np.sin(self._stm.phases), np.cos(self._stm.phases)))
        return obs

    def makeRealWorldObs(self, obs=None, noisy=False):
        if obs is None:
            obs = RealWorldObservation()
        obs = self._setObsBaseAttr(obs, 'noisy_proprio' if noisy else 'proprio')
        r = self._robot
        obs.joint_prev_pos_err = r.getJointPosErrHistoryFromIndex(-1, noisy)
        obs.ftg_frequencies = self._stm.frequency
        obs.ftg_phases = np.concatenate((np.sin(self._stm.phases), np.cos(self._stm.phases)))
        obs.joint_pos_err_his = np.concatenate((r.getJointPosErrHistoryFromMoment(-0.01, noisy),
                                                r.getJointPosErrHistoryFromMoment(-0.02, noisy)))
        obs.joint_vel_his = np.concatenate((r.getJointVelHistoryFromMoment(-0.01, noisy),
                                            r.getJointVelHistoryFromMoment(-0.02, noisy)))
        if not self._action_history:
            obs.joint_pos_target = obs.joint_prev_pos_target = r.STANCE_POSTURE
        else:
            obs.joint_pos_target = self._action_history[-1]
            if len(self._action_history) > 1:
                obs.joint_prev_pos_target = self._action_history[-2]
            else:
                obs.joint_prev_pos_target = self.robot.STANCE_POSTURE
        obs.base_frequency = (self._stm.base_frequency,)
        return obs

    def makeExtendedObs(self, obs=None, noisy=False):
        if obs is None:
            obs = ExtendedObservation()
        obs: ExtendedObservation = self._setObsBaseAttr(obs, 'noisy_realworld' if noisy else 'realworld')
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

    def step(self, action: Union[Action, np.ndarray]):
        if not isinstance(action, Action):
            action = Action.from_array(action)
        self._stm.update((0.,) * 4)
        # self._stm.update(action.leg_frequencies)
        priori = self._stm.get_priori_trajectory().reshape(4, 3)
        des_pos = action.foot_pos_residuals.reshape(4, 3) + priori

        if g_cfg.plot_trajectory:
            self.plotFootTrajectories(des_pos)
        return super().step(des_pos)

    def reset(self):
        self._stm.reset()
        return super().reset()

    def plotFootTrajectories(self, des_pos):
        from burl.utils import plot_trajectory
        if not hasattr(self, '_plotter'):
            self._plotter = plot_trajectory()
        for i, flag in enumerate(self._stm.cycles == 5):
            if flag:
                x, y, z = des_pos[i]
                self._plotter(i, (x, z), 'r')
                x, y, z = self._robot.getFootPositionInHipFrame(i)
                self._plotter(i, (x, z), 'b')

    def _prepareSimulation(self):  # for the stability of the beginning
        for _ in range(100):
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
    g_cfg.extra_visualization = True
    g_cfg.test_profile()
    # g_cfg.slow_down_rendering()
    # g_cfg.motor_latencies = (2e-3, 0.)
    init_logger()
    set_logger_level('DEBUG')
    np.set_printoptions(precision=3, linewidth=1000)
    tg = True
    if tg:
        env = FixedTgEnv(robot_auto_maker())
        env.initObservation()
        for i in range(1, 10000):
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
