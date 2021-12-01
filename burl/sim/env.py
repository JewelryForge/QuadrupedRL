import time
from collections import deque
from itertools import chain

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from burl.sim.motor import MotorSim
from burl.sim.quadruped import A1, Quadruped
from burl.sim.terrain import PlainTerrain, RandomUniformTerrain
from burl.rl.state import ExtendedObservation, Action
from burl.rl.tg import LocomotionStateMachine
from burl.rl.task import BasicTask
from burl.utils import make_cls, g_cfg, logger, set_logger_level
from burl.utils.transforms import Rpy
import pkgutil


class QuadrupedEnv(object):
    def __init__(self, make_robot=A1, make_task=BasicTask):
        self._gui = g_cfg.rendering_enabled
        self._env = bullet_client.BulletClient(pybullet.GUI if self._gui else pybullet.DIRECT) if True else pybullet
        self._env.setAdditionalSearchPath(pybullet_data.getDataPath())
        if g_cfg.plain:
            self._terrain = PlainTerrain(self._env)
        else:
            self._terrain = RandomUniformTerrain(self._env, size=g_cfg.trn_size, downsample=g_cfg.trn_downsample,
                                                 resolution=g_cfg.trn_resolution, offset=g_cfg.trn_offset, seed=2)
        # self._loadEgl()
        if self._gui:
            self._prepareRendering()
        self._robot: Quadruped = make_robot(sim_env=self._env)
        self._task = make_task(self)
        assert g_cfg.sim_frequency >= g_cfg.execution_frequency >= g_cfg.action_frequency

        self._setPhysicsParameters()
        self._num_action_repeats = int(g_cfg.sim_frequency / g_cfg.action_frequency)
        self._num_execution_repeats = int(g_cfg.sim_frequency / g_cfg.execution_frequency)
        logger.debug(f'Action Repeats for {self._num_action_repeats} time(s)')
        logger.debug(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        if self._gui:
            self._initRendering()
        self._sim_step_counter = 0
        self._action_buffer = deque(maxlen=10)

    @property
    def robot(self):
        return self._robot

    @property
    def terrain(self):
        return self._terrain

    def initObservation(self):
        self.updateObservation()
        return self.makeStandardObservation(True), self.makeStandardObservation(False)

    def updateObservation(self):
        return self._robot.updateObservation()

    def _loadEgl(self):
        if egl := pkgutil.get_loader('eglRenderer'):
            print(egl.get_filename())
            self._env.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
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

        self._dbg_reset = self._env.addUserDebugParameter('reset', 1, 0, 0)
        self._reset_counter = 0

        self._last_frame_time = time.time()
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, True)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, True)

    def _setPhysicsParameters(self):
        # self._env.resetSimulation()
        # self._env.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
        self._env.setTimeStep(1 / g_cfg.sim_frequency)
        self._env.setGravity(0, 0, -9.8)

    def _updateRendering(self):
        if (current := self._env.readUserDebugParameter(self._dbg_reset)) != self._reset_counter:
            self._reset_counter = current
            self.reset()
        if g_cfg.sleeping_enabled:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._num_action_repeats / g_cfg.sim_frequency - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        if g_cfg.moving_camera:
            yaw, pitch, dist = self._env.getDebugVisualizerCamera()[8:11]
            self._env.resetDebugVisualizerCamera(dist, yaw, pitch, self._robot.position)
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

            r = self._robot
            foot_xy = [r.getFootPositionInWorldFrame(i)[:2] for i in range(4)]
            positions = chain(*[self.getAbundantTerrainInfo(x, y, r.orientation) for x, y in foot_xy])
            for idc, pos in zip(self._terrain_indicators, positions):
                self._env.resetBasePositionAndOrientation(idc, posObj=pos, ornObj=(0, 0, 0, 1))

    def makeStandardObservation(self, privileged=True):
        eo = ExtendedObservation()
        r = self._robot
        if_noisy = not privileged
        eo.command = self._task.cmd
        eo.gravity_vector = r.getBaseAxisZ()
        eo.base_linear = r.getBaseLinearVelocityInBaseFrame(if_noisy)
        eo.base_angular = r.getBaseAngularVelocityInBaseFrame(if_noisy)
        eo.joint_pos = r.getJointPositions(if_noisy)
        eo.joint_vel = r.getJointVelocities(if_noisy)
        eo.joint_prev_pos_err = r.getJointPosErrHistoryFromIndex(-1, if_noisy)
        eo.joint_pos_err_his = np.concatenate((r.getJointPosErrHistoryFromMoment(-0.01, if_noisy),
                                               r.getJointPosErrHistoryFromMoment(-0.02, if_noisy)))
        eo.joint_vel_his = np.concatenate((r.getJointVelHistoryFromMoment(-0.01, if_noisy),
                                           r.getJointVelHistoryFromMoment(-0.02, if_noisy)))
        eo.joint_pos_target = r.getCmdHistoryFromIndex(-1)
        eo.joint_prev_pos_target = r.getCmdHistoryFromIndex(-2)
        foot_xy = [r.getFootPositionInWorldFrame(i)[:2] for i in range(4)]
        eo.terrain_scan = np.concatenate([self.getTerrainScan(x, y, r.orientation) for x, y in foot_xy])
        eo.terrain_normal = np.concatenate([self.getTerrainNormal(x, y) for x, y in foot_xy])
        eo.contact_states = r.getContactStates()[1:]
        eo.foot_contact_forces = r.getFootContactForces()
        eo.foot_friction_coeffs = [self.getTerrainFrictionCoeff(x, y) for x, y in foot_xy]
        eo.external_disturbance = r.getBaseDisturbance()
        return eo

    def step(self, action):
        # NOTICE: ADDING LATENCY ARBITRARILY FROM A DISTRIBUTION IS NOT REASONABLE
        # NOTICE: SHOULD CALCULATE TIME_SPENT IN REAL WORLD; HERE USE FIXED TIME INTERVAL
        # start = time.time()
        rewards = []
        self._action_buffer.append(action)
        for _ in range(self._num_action_repeats):
            update_execution = self._sim_step_counter % self._num_execution_repeats == 0
            if update_execution:
                torques = self._robot.applyCommand(action)
            self._sim_step_counter += 1
            self._env.stepSimulation()
            rewards.append(self._task.calculateReward())
            if update_execution:
                self._robot.updateObservation()
        if self._gui:
            self._updateRendering()
        time_out = self._sim_step_counter >= g_cfg.max_sim_iterations
        info = {'time_out': time_out, 'torques': torques, 'reward_details': self._task.getRewardDetails()}
        # logger.debug(f'Step time: {time.time() - start}')
        return (self.makeStandardObservation(True),
                self.makeStandardObservation(False),
                np.mean(rewards),
                (not self._robot.is_safe()) or self._task.done() or time_out,
                info)

    def reset(self, **kwargs):
        completely_reset = kwargs.get('completely_reset', False)
        self._sim_step_counter = 0
        self._task.reset()
        if completely_reset:
            raise NotImplementedError
        return self._robot.reset()

    def close(self):
        self._env.disconnect()

    def getActionMutation(self):
        if len(self._action_buffer) < 3:
            return 0.0
        actions = [self._action_buffer[-i - 1] for i in range(3)]
        return np.linalg.norm(actions[0] - 2 * actions[1] + actions[2]) * g_cfg.action_frequency ** 2

    def getAbundantTerrainInfo(self, x, y, orientation):
        interval = 0.1
        yaw = Rpy.from_quaternion(orientation).y
        dx, dy = interval * np.cos(yaw), interval * np.sin(yaw)
        points = ((dx - dy, dx + dy), (dx, dy), (dx + dy, -dx + dy),
                  (-dy, dx), (0, 0), (dy, -dx),
                  (-dx - dy, dx - dy), (-dx, -dy), (-dx + dy, -dx - dy))
        return [(xp := x + dx, yp := y + dy, self.getTerrainHeight(xp, yp)) for dx, dy in points]

    def getTerrainScan(self, x, y, orientation):
        return [p[2] for p in self.getAbundantTerrainInfo(x, y, orientation)]

    def getTerrainHeight(self, x, y):
        return self._terrain.getHeight(x, y)

    def getTerrainNormal(self, x, y):
        return self._terrain.getNormal(x, y)

    def getTerrainFrictionCoeff(self, x, y):
        return 0.0


class TGEnv(QuadrupedEnv):
    def __init__(self, **kwargs):
        super(TGEnv, self).__init__(**kwargs)
        self._stm = LocomotionStateMachine(1 / g_cfg.action_frequency)

    def makeStandardObservation(self, privileged=True):
        eo: ExtendedObservation = super().makeStandardObservation(privileged)
        eo.ftg_frequencies = self._stm.frequency
        eo.ftg_phases = np.concatenate((np.sin(self._stm.phases), np.cos(self._stm.phases)))
        logger.debug(eo.__dict__)
        return (eo.to_array() - eo.offset) * eo.scale

    def step(self, action: Action):
        # 0 ~ 3 additional frequencies
        # 4 ~ 11 foot position residual
        self._stm.update(action.leg_frequencies)
        priori = self._stm.get_priori_trajectory() - self.robot.STANCE_HEIGHT
        if False:
            h2b = self._robot.getHorizontalFrameInBaseFrame(False)  # FIXME: HERE SHOULD BE TRUE
            priori_in_base_frame = [h2b @ (0, 0, z) for z in priori]
            residuals = action.foot_pos_residuals + np.concatenate(priori_in_base_frame)
        else:
            residuals = action.foot_pos_residuals
            for i in range(4):
                residuals[i * 3 + 2] += priori[i]

        # NOTICE: HIP IS USED IN PAPER
        commands = [self._robot.ik(i, residuals[i * 3: i * 3 + 3], 'shoulder') for i in range(4)]
        # TODO: COMPLETE NOISY OBSERVATION CONVERSIONS
        return super().step(np.concatenate(commands))

    def reset(self, **kwargs):
        self._stm.reset()
        return super().reset()


if __name__ == '__main__':
    g_cfg.moving_camera = False
    g_cfg.sleeping_enabled = True
    g_cfg.on_rack = False
    g_cfg.rendering_enabled = True
    g_cfg.plain = True
    set_logger_level(logger.DEBUG)
    np.set_printoptions(precision=2, linewidth=1000)
    make_motor = make_cls(MotorSim)
    env = QuadrupedEnv()
    # env = TGEnv()
    robot = env.robot
    env.initObservation()
    for _ in range(100000):
        # act = Action()
        # env.step(act)
        cmd0 = robot.ik(0, (0, 0, -0.3), 'shoulder')
        cmd1 = robot.ik(1, (0, 0, -0.3), 'shoulder')
        cmd2 = robot.ik(2, (0, 0, -0.3), 'shoulder')
        cmd3 = robot.ik(3, (0, 0, -0.3), 'shoulder')
        env.step((np.concatenate([cmd0, cmd1, cmd2, cmd3])))
