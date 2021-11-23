import time

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

from burl.rl.state import ExtendedObservation
from burl.sim.motor import MotorSim
from burl.sim.quadruped import A1, Quadruped
from burl.sim.terrain import make_plane
from burl.rl.tg import LocomotionStateMachine
from burl.utils import make_cls, RenderParam, SimParam, PhysicsParam
from burl.rl.task import BasicTask
from burl.utils.transforms import Rpy


class QuadrupedEnv(object):
    def __init__(self, make_robot=A1, make_task=BasicTask,
                 sim_param=SimParam(), render_param=RenderParam()):
        self._cfg, self._render_cfg = sim_param, render_param
        self._gui = render_param.rendering_enabled
        if self._gui:
            self._env = bullet_client.BulletClient(pybullet.GUI)
            self._initRendering()
        else:
            self._env = bullet_client.BulletClient(pybullet.DIRECT)
        if False:
            self._env = pybullet

        self._env.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._robot: Quadruped = make_robot(sim_env=self._env, frequency=sim_param.execution_frequency)
        self._task = make_task(self)
        # TODO: RANDOMLY CHANGE TERRAIN
        self._terrain_generator = make_plane
        self._sim_frequency = sim_param.sim_frequency
        self._action_frequency = sim_param.action_frequency
        assert self._sim_frequency >= sim_param.execution_frequency >= self._action_frequency

        self._terrain = self._terrain_generator(self._env)
        self._setPhysicsParameters()
        self._num_action_repeats = int(self._sim_frequency / self._action_frequency)
        self._num_execution_repeats = int(self._sim_frequency / sim_param.execution_frequency)
        print(f'Action Repeats for {self._num_action_repeats} time(s)')
        print(f'Execution Repeats For {self._num_execution_repeats} time(s)')
        self._sim_step_counter = 0

    @property
    def robot(self):
        return self._robot

    def initObservation(self):
        self.updateObservation()
        return self.makeStandardObservation(True), self.makeStandardObservation(False)

    def updateObservation(self):
        return self._robot.updateObservation()

    def _initRendering(self):
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, True)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, True)
        # if hasattr(self._task, '_draw_ref_model_alpha'):
        #     self._show_reference_id = pybullet.addUserDebugParameter("show reference", 0, 1,
        #                                                              self._task._draw_ref_model_alpha)
        self._dbg_reset = self._env.addUserDebugParameter('reset', 1, 0, 0)
        self._reset_counter = 0

        if self._render_cfg.egl_rendering:  # TODO: WHAT DOES THE PLUGIN DO?
            self._env.loadPlugin('eglRendererPlugin')
        self._last_frame_time = time.time()

    def _setPhysicsParameters(self):
        # self._env.resetSimulation()
        # self._env.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
        self._env.setTimeStep(1 / self._sim_frequency)
        self._env.setGravity(0, 0, -9.8)

    def _updateRendering(self):
        if (current := self._env.readUserDebugParameter(self._dbg_reset)) != self._reset_counter:
            self._reset_counter = current
            self.reset()
        if self._render_cfg.sleeping_enabled:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._num_action_repeats / self._sim_frequency - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        # Keep the previous orientation of the camera set by the user.
        yaw, pitch, dist = self._env.getDebugVisualizerCamera()[8:11]
        self._env.resetDebugVisualizerCamera(dist, yaw, pitch, self._robot.position)
        self._env.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

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
        eo.terrain_scan = np.concatenate([self.getTerrainScan(*xy, r.orientation) for xy in foot_xy])
        eo.terrain_normal = np.concatenate([self.getTerrainNormal(*xy) for xy in foot_xy])
        eo.contact_states = r.getContactStates()[1:]
        eo.foot_contact_forces = r.getFootContactForces()
        eo.foot_friction_coeffs = [self.getTerrainFrictionCoeff(*xy) for xy in foot_xy]
        eo.external_disturbance = r.getBaseDisturbance()
        return eo

    def step(self, action):
        # NOTICE: ADDING LATENCY ARBITRARILY FROM A DISTRIBUTION IS NOT REASONABLE
        # NOTICE: SHOULD CALCULATE TIME_SPENT IN REAL WORLD; HERE USE FIXED TIME INTERVAL
        for _ in range(self._num_action_repeats):
            update_execution = self._sim_step_counter % self._num_execution_repeats == 0
            if update_execution:
                self._robot.applyCommand(action)
            self._sim_step_counter += 1
            self._env.stepSimulation()
            if update_execution:
                self._robot.updateObservation()
        if self._gui:
            self._updateRendering()
        return (self.makeStandardObservation(True), self.makeStandardObservation(False),
                self._task.calculateReward(), self._robot.is_safe() or self._task.done(), {})

    def reset(self, **kwargs):
        completely_reset = kwargs.get('completely_reset', False)
        if completely_reset:
            raise NotImplementedError
        return self._robot.reset()

    def close(self):
        self._env.disconnect()

    def getTerrainScan(self, x, y, orientation):
        interval = 0.1
        yaw = Rpy.from_quaternion(orientation).y
        dx, dy = interval * np.cos(yaw), interval * np.sin(yaw)
        points = ((dx - dy, dx + dy), (dx, dy), (dx + dy, -dx + dy),
                  (-dy, dx), (0, 0), (dy, -dx),
                  (-dx - dy, dx - dy), (-dx, -dy), (-dx + dy, -dx - dy))
        return [self.getTerrainHeight(x + p[0], y + p[1]) for p in points]

    def getTerrainHeight(self, x, y):
        return 0.0

    def getTerrainNormal(self, x, y):
        return (0., 0., 1.)

    def getTerrainFrictionCoeff(self, x, y):
        return 0.0


class TGEnv(QuadrupedEnv):
    def __init__(self, **kwargs):
        super(TGEnv, self).__init__(**kwargs)
        self._stm = LocomotionStateMachine(1 / self._action_frequency)

    def makeStandardObservation(self, privileged=True):
        eo: ExtendedObservation = super().makeStandardObservation(privileged)
        eo.ftg_frequencies = self._stm.frequency
        eo.ftg_phases = np.concatenate((np.sin(self._stm.phases), np.cos(self._stm.phases)))
        # print(eo.__dict__)
        return (eo.to_array() - eo.offset) * eo.scale

    def step(self, action):
        # 0 ~ 3 additional frequencies
        # 4 ~ 11 foot position residual
        self._stm.update(action[:4])
        priori = self._stm.get_priori_trajectory() - self.robot.STANCE_HEIGHT
        residuals = action[4:]
        for i in range(4):
            residuals[i * 3 + 2] += priori[i]
        # NOTICE: HIP IS USED IN PAPER
        commands = [self._robot.ik(i, residuals[i * 3: i * 3 + 3], 'shoulder') for i in range(4)]
        # TODO: COMPLETE NOISY OBSERVATION CONVERSIONS
        return super().step(np.concatenate(commands))


# if __name__ == '__main__':
#     np.set_printoptions(precision=2, linewidth=1000)
#     make_motor = make_cls(MotorSim)
#     physics_param = PhysicsParam()
#     physics_param.on_rack = False
#     env = QuadrupedEnv()
#     env.updateObservation()
#     for _ in range(100000):
#         time.sleep(1. / 240.)
#         cmd0 = env.robot.ik(0, (0, 0, -0.3), 'shoulder')
#         cmd1 = env.robot.ik(1, (0, 0, -0.3), 'shoulder')
#         cmd2 = env.robot.ik(2, (0, 0, -0.3), 'shoulder')
#         cmd3 = env.robot.ik(3, (0, 0, -0.3), 'shoulder')
#         env.step(np.concatenate([cmd0, cmd1, cmd2, cmd3]))

if __name__ == '__main__':
    np.set_printoptions(precision=2, linewidth=1000)
    make_motor = make_cls(MotorSim)
    physics_param = PhysicsParam()
    physics_param.on_rack = False
    env = TGEnv()
    env.initObservation()
    for _ in range(100000):
        time.sleep(1. / 240.)
        env.step([0.] * 16)
