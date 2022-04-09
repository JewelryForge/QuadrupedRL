import math
import os.path
import time
from itertools import chain
from typing import Union

import numpy as np
import pybullet as pyb

from burl.utils import UdpPublisher, Angle, unit, vec_cross, sign, log_info

__all__ = ['Plugin', 'StatisticsCollector', 'InfoRenderer', 'VideoRecorder']

from burl.utils.transforms import Rpy


class Plugin(object):
    utils = []

    def on_init(self, task, robot, env):
        pass

    def on_sim_step(self, task, robot, env):
        pass

    def on_step(self, task, robot, env) -> Union[dict, None]:
        pass

    def on_reset(self, task, robot, env):
        pass


class StatisticsCollector(Plugin):
    def __init__(self, publish=True):
        self._torque_sum = 0.
        self._torque_abs_sum = 0.
        self._torque_pen_sum = 0.0
        self._joint_motion_sum = 0.0
        self._sim_step_counter = 0
        self._step_counter = 0
        self._publish = publish
        if self._publish:
            self._udp_pub = UdpPublisher(9870)

    def on_sim_step(self, task, robot, env):
        from burl.sim.env import Quadruped, FixedTgEnv
        from burl.rl.reward import TorquePenalty, JointMotionPenalty, OrthogonalLinearPenalty

        cmd = task.cmd
        env: FixedTgEnv
        rob: Quadruped = robot

        def wrap(reward_type):
            return reward_type()(cmd, env, rob)

        self._sim_step_counter += 1
        self._torque_sum += rob.getLastAppliedTorques() ** 2
        self._torque_abs_sum += abs(rob.getLastAppliedTorques())
        self._torque_pen_sum += wrap(TorquePenalty)
        self._joint_motion_sum += wrap(JointMotionPenalty)
        # print(wrap(OrthogonalLinearPenalty))
        # print(wrap(LinearVelocityReward))
        # print(rob.getJointVelocities())
        # print(rob.getJointAccelerations())
        # print()
        # print(max(rob.getLastAppliedTorques()))
        # print(wrap(HipAnglePenalty))
        # print(rob.getBaseLinearVelocityInBaseFrame()[2])

        # print(wrap(TorquePenalty))
        # r_rate, p_rate, _ = rob.getBaseRpyRate()
        # print(r_rate, p_rate, wrap(RollPitchRatePenalty))
        # r, p, _ = rob.rpy
        # print(r, p, wrap(BodyPosturePenalty))
        # print(cmd, rob.getBaseLinearVelocityInBaseFrame()[:2], wrap(LinearVelocityReward))
        # print(env.getSafetyHeightOfRobot(), wrap(BodyHeightReward))
        # print(rob.getCostOfTransport(), wrap(CostOfTransportReward))
        # strides = [np.linalg.norm(s) for s in rob.getStrides()]
        # if any(s != 0.0 for s in strides):
        #     print(strides, wrap(SmallStridePenalty))
        # if any(clearances := rob.getFootClearances()):
        #     print(clearances, wrap(FootClearanceReward))
        if self._publish:
            data = {
                'joint_states': {
                    'joint_pos': (rob.getJointPositions() - rob.STANCE_POSTURE).tolist(),
                    'commands': rob.getLastCommand().tolist(),
                    'joint_vel': rob.getJointVelocities().tolist(),
                    'joint_acc': rob.getJointAccelerations().tolist(),
                    # 'kp_part': tuple(rob._motor._kp_part),
                    # 'kd_part': tuple(rob._motor._kd_part),
                    'torque': rob.getLastAppliedTorques().tolist(),
                    'contact': rob.getContactStates().tolist()
                },
                'body_height': env.getTerrainBasedHeightOfRobot(),
                'cot': rob.getCostOfTransport(),
                'twist': {
                    'linear': rob.getBaseLinearVelocityInBaseFrame().tolist(),
                    'angular': rob.getBaseAngularVelocityInBaseFrame().tolist(),
                },
                'torque_pen': wrap(TorquePenalty)
            }
            self._udp_pub.send(data)

    def on_step(self, task, robot, env):
        self._step_counter += 1

    def on_reset(self, task, robot, env):
        print('episode len:', self._step_counter)
        print('cot', robot.getCostOfTransport())
        print('mse torque', np.sqrt(self._torque_sum / self._sim_step_counter))
        print('abs torque', self._torque_abs_sum / self._sim_step_counter)
        print('torque pen', self._torque_pen_sum / self._sim_step_counter)
        print('joint motion pen', self._joint_motion_sum / self._sim_step_counter)
        self._torque_sum = 0.
        self._torque_abs_sum = 0.
        self._torque_pen_sum = 0.0
        self._joint_motion_sum = 0.0
        self._sim_step_counter = 0
        self._step_counter = 0


class InfoRenderer(Plugin):
    def __init__(self, extra_vis=True, show_time_ratio=True, show_indicators=True, driving_mode=True,
                 moving_camera=False, allow_sleeping=True, single_step_rendering=False):
        self.extra_vis, self.show_time_ratio, self.show_indicators = extra_vis, show_time_ratio, show_indicators
        self.driving_mode, self.sleeping, self.single = driving_mode, allow_sleeping, single_step_rendering
        self.moving_camera = driving_mode or moving_camera
        if self.extra_vis:
            self._contact_visual_shape = self._terrain_visual_shape = -1
            self._contact_obj_ids = []
            self._terrain_indicators = None

        if self.show_indicators:
            self._force_indicator = self._torque_indicator = -1
            self._cmd_indicators = [-1] * 7
            self._tip_axis_x = self._tip_axis_y = self._tip_last_end = None  # torque indicator plane
            self._tip_phase = 0
            self._external_force_buffer = self._external_torque_buffer = self._cmd_buffer = None

        if self.show_time_ratio:
            self._last_time_ratio = 0.
            self._time_ratio_indicator = -1

        if self.driving_mode:
            self._robot_yaw_filter = []

        self._last_frame_time = 0.

    def on_step(self, task, robot, env):
        if not self.single:
            self.update_rendering(task, robot, env)

    def on_sim_step(self, task, robot, env):
        if self.single:
            self.update_rendering(task, robot, env)
            env.client.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, True)

    def on_init(self, task, robot, env):
        sim_env = env.client
        if self.extra_vis:
            self._contact_visual_shape = sim_env.createVisualShape(
                shapeType=pyb.GEOM_BOX, halfExtents=(0.03, 0.03, 0.03), rgbaColor=(0.8, 0., 0., 0.6))
            self._terrain_visual_shape = sim_env.createVisualShape(
                shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0., 0.8, 0., 0.6))
            self._terrain_indicators = [sim_env.createMultiBody(baseVisualShapeIndex=self._terrain_visual_shape)
                                        for _ in range(36)]
        if self.moving_camera and not self.driving_mode:
            sim_env.resetDebugVisualizerCamera(
                1.5, 0., 0., (0., 0., robot.STANCE_HEIGHT + env.getTerrainHeight(0., 0.)))

        if self.driving_mode:
            (x, y, _), orn = pyb.getBasePositionAndOrientation(robot.id)
            z = robot.STANCE_HEIGHT + env.getTerrainHeight(x, y)
            yaw = (Rpy.from_quaternion(orn).y - math.pi / 2) / math.pi * 180
            sim_env.resetDebugVisualizerCamera(1.5, yaw, -30., (x, y, z))
        elif self.moving_camera:
            sim_env.resetDebugVisualizerCamera(
                1.5, 0., 0., (0., 0., robot.STANCE_HEIGHT + env.getTerrainHeight(0., 0.)))
        self._last_frame_time = time.time()

    def on_reset(self, task, robot, env):
        if self.driving_mode:
            self._robot_yaw_filter.clear()

    def update_rendering(self, task, robot, env):
        sim_env = env.client
        time_spent = time.time() - self._last_frame_time
        freq = env.sim_freq if self.single else env.action_freq
        if self.sleeping:
            period = 1 / freq
            if (time_to_sleep := period - time_spent) > 0:
                time.sleep(time_to_sleep)
                time_spent += time_to_sleep
        self._last_frame_time = time.time()
        time_ratio = 1 / freq / time_spent
        if self.show_time_ratio:
            if self._last_time_ratio != time_ratio:
                _time_ratio_indicator = sim_env.addUserDebugText(
                    f'{time_ratio: .2f}', textPosition=(0., 0., 0.), textColorRGB=(1., 1., 1.),
                    textSize=1, lifeTime=0, parentObjectUniqueId=robot.id,
                    replaceItemUniqueId=self._time_ratio_indicator)
                if self._time_ratio_indicator != -1 and _time_ratio_indicator != self._time_ratio_indicator:
                    sim_env.removeUserDebugItem(self._time_ratio_indicator)
                self._time_ratio_indicator = _time_ratio_indicator
                self._last_time_ratio = time_ratio

        if self.moving_camera:
            x, y, _ = robot.position
            z = robot.STANCE_HEIGHT + env.getTerrainHeight(x, y)
            if self.driving_mode:
                self._robot_yaw_filter.append(robot.rpy.y - math.pi / 2)
                if len(self._robot_yaw_filter) > 100:
                    self._robot_yaw_filter = self._robot_yaw_filter[-10:]
                # To avoid carsick :)
                mean = Angle.mean(self._robot_yaw_filter[-10:]) / math.pi * 180
                sim_env.resetDebugVisualizerCamera(1.5, mean, -30., (x, y, z))
            else:
                yaw, pitch, dist = sim_env.getDebugVisualizerCamera()[8:11]
                sim_env.resetDebugVisualizerCamera(dist, yaw, pitch, (x, y, z))

        if self.extra_vis:
            for obj in self._contact_obj_ids:
                sim_env.removeBody(obj)
            self._contact_obj_ids.clear()
            for cp in sim_env.getContactPoints(bodyA=robot.id):
                pos, normal, normal_force = cp[5], cp[7], cp[9]
                if normal_force > 0.1:
                    obj = sim_env.createMultiBody(baseVisualShapeIndex=self._contact_visual_shape,
                                                  basePosition=pos)
                    self._contact_obj_ids.append(obj)
            positions = chain(*[env.getAbundantTerrainInfo(x, y, robot.rpy.y)
                                for x, y in robot.getFootXYsInWorldFrame()])
            for idc, pos in zip(self._terrain_indicators, positions):
                sim_env.resetBasePositionAndOrientation(idc, posObj=pos, ornObj=(0, 0, 0, 1))
        if self.show_indicators:
            external_force, external_torque = env.getDisturbance()
            if self._external_force_buffer is not external_force:
                _force_indicator = sim_env.addUserDebugLine(
                    lineFromXYZ=(0., 0., 0.), lineToXYZ=external_force / 50, lineColorRGB=(1., 0., 0.),
                    lineWidth=3, lifeTime=1,
                    parentObjectUniqueId=robot.id,
                    replaceItemUniqueId=self._force_indicator)
                if self._force_indicator != -1 and _force_indicator != self._force_indicator:
                    sim_env.removeUserDebugItem(self._force_indicator)
                self._force_indicator = _force_indicator
                self._external_force_buffer = external_force

            if (external_torque != 0).any():
                magnitude = math.hypot(*external_torque)
                axis_z = external_torque / magnitude
                assis = np.array((0., 0., 1.) if any(axis_z != (0., 0., 1.)) else (1., 0., 0.))
                self._tip_axis_x = unit(vec_cross(axis_z, assis))
                self._tip_axis_y = vec_cross(axis_z, self._tip_axis_x)
                if self._tip_last_end is None:
                    self._tip_last_end = self._tip_axis_x * magnitude / 20
                self._tip_phase += math.pi / 36
                tip_end = (self._tip_axis_x * math.cos(self._tip_phase) +
                           self._tip_axis_y * math.sin(self._tip_phase)) * magnitude / 20
                _torque_indicator = sim_env.addUserDebugLine(
                    lineFromXYZ=self._tip_last_end, lineToXYZ=tip_end, lineColorRGB=(0., 0., 1.),
                    lineWidth=5, lifeTime=0.1,
                    parentObjectUniqueId=robot.id,
                    replaceItemUniqueId=self._torque_indicator)
                self._tip_last_end = tip_end

            if self._cmd_buffer is not (cmd := task.cmd.copy()):
                if ((linear := cmd[:2]) != 0.).any():
                    axis_x, axis_y = np.array((*linear, 0)), np.array((-linear[1], linear[0], 0))
                    last_end = axis_x * 0.1
                    phase, phase_inc = 0, cmd[2] / 3
                    # plot arrow ----->
                    for i, cmd_idc in enumerate(self._cmd_indicators):
                        if i < 5:
                            phase += phase_inc
                            inc = (axis_x * math.cos(phase) + axis_y * math.sin(phase)) / 15
                        else:
                            phase += math.pi / 12 if i == 5 else -math.pi / 6
                            inc = -(axis_x * math.cos(phase) + axis_y * math.sin(phase)) / 15
                        end = last_end + inc
                        _cmd_indicator = sim_env.addUserDebugLine(
                            lineFromXYZ=last_end, lineToXYZ=end, lineColorRGB=(1., 1., 0.),
                            lineWidth=5, lifeTime=1, parentObjectUniqueId=robot.id,
                            replaceItemUniqueId=cmd_idc)
                        self._cmd_indicators[i] = _cmd_indicator
                        if i < 5:
                            last_end = end
                else:
                    phase, phase_inc = 0, cmd[2] / 3
                    radius = abs(cmd[2] / 6)
                    last_end = (radius, 0., 0.)
                    for i, cmd_idc in enumerate(self._cmd_indicators):
                        if i < 5:
                            phase += phase_inc
                            end = np.array((math.cos(phase), math.sin(phase), 0.)) * radius
                        else:
                            if i == 5:
                                _phase = phase + (-math.pi / 2 + math.pi / 12) * sign(cmd[2])
                            else:
                                _phase = phase + (-math.pi / 2 - math.pi / 4) * sign(cmd[2])
                            length = radius * math.sin(abs(phase_inc) / 2) * 3
                            end = last_end + np.array((math.cos(_phase), math.sin(_phase), 0.)) * length
                        _cmd_indicator = sim_env.addUserDebugLine(
                            lineFromXYZ=last_end, lineToXYZ=end, lineColorRGB=(1., 1., 0.),
                            lineWidth=5, lifeTime=1, parentObjectUniqueId=robot.id,
                            replaceItemUniqueId=cmd_idc)
                        self._cmd_indicators[i] = _cmd_indicator
                        if i < 5:
                            last_end = end


class CameraImageRecorder(Plugin):
    def __init__(self, fps, size=(1024, 768), dst='', single_step_rendering=False):
        try:
            import cv2
            self.size = size
            self.enabled = True
            self.single = single_step_rendering
            if not dst:
                from burl.exp import g_log_dir, get_timestamp
                dst = os.path.join(g_log_dir, f'record-{get_timestamp()}.avi')
            self.writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
        except ModuleNotFoundError:
            self.enabled = False
            print('`opencv-python` is required for recording')

    @classmethod
    def is_available(cls):
        try:
            import cv2
            return True
        except ModuleNotFoundError:
            return False

    def on_init(self, task, robot, env):
        env.client.configureDebugVisualizer(pyb.COV_ENABLE_RGB_BUFFER_PREVIEW, True)

    def on_step(self, task, robot, env):
        if self.enabled and not self.single:
            self.write_camera_image(env.client)

    def on_sim_step(self, task, robot, env):
        if self.enabled and self.single:
            self.write_camera_image(env.client)

    def on_reset(self, task, robot, env):
        self.writer.release()
        self.enabled = False

    def write_camera_image(self, sim_env):
        _, _, view_mat, proj_mat, *_ = sim_env.getDebugVisualizerCamera()
        _, _, rgba, *_ = sim_env.getCameraImage(*self.size, view_mat, proj_mat,
                                                renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
        bgr = rgba[..., 2::-1]
        self.writer.write(bgr)


class VideoRecorder(Plugin):
    def __init__(self, dst=''):
        if not dst:
            from burl.exp import g_log_dir, get_timestamp
            dst = os.path.join(g_log_dir, f'record-{get_timestamp()}.mp4')
        self.dst = dst
        self.log_id = -1

    def on_init(self, task, robot, env):
        log_info('start recording')
        self.log_id = env.client.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4, self.dst)

    def on_reset(self, task, robot, env):
        if self.log_id != -1:
            env.client.stopStateLogging(self.log_id)
            self.log_id = -1
