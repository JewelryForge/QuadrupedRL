import itertools
import time

import numpy as np
import pybullet as pyb
import pybullet_data

from qdpgym.sim.blt.env import QuadrupedEnv
from qdpgym.sim.blt.hooks import ViewerHook
from qdpgym.sim.blt.quadruped import Aliengo
from qdpgym.sim.blt.terrain import Plain, Hills, Slopes, Steps, PlainHf
from qdpgym.sim.task import NullTask
from qdpgym.tasks.loct import LocomotionV0, LocomotionPMTG
from qdpgym.utils import tf


def test_robot():
    pyb.connect(pyb.GUI)
    pyb.setTimeStep(2e-3)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    Plain().spawn(pyb)
    rob = Aliengo(500, 'pd')
    rs = np.random.RandomState()
    rob.spawn(pyb, rs)
    pyb.setGravity(0, 0, -9.8)

    for _ in range(1000):
        # with MfTimer() as t:
        pyb.stepSimulation()
        rob.update_observation(rs)
        rob.apply_command(rob.STANCE_CONFIG)

    pyb.disconnect()

def test_robot_extent():
    pyb.connect(pyb.GUI)
    pyb.setTimeStep(2e-3)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    Plain().spawn(pyb)
    rob = Aliengo(500, 'pd')
    rob.spawn_on_rack(pyb, np.random)
    pyb.setGravity(0, 0, -9.8)

    joint_pos = np.concatenate((
        rob.inverse_kinematics(0, (0.3, 0., 0.05)),
        rob.inverse_kinematics(0, (0.35, 0., 0.1)),
        rob.inverse_kinematics(0, (0.25, 0., 0.)),
        rob.inverse_kinematics(0, (-0.25, 0., 0.)),
    ))

    for _ in range(1000):
        pyb.stepSimulation()
        rob.update_observation(np.random)

        rob.apply_command(joint_pos)
        time.sleep(0.002)

    pyb.disconnect()


def test_env():
    rob = Aliengo(500, 'pd')
    arena = Plain()
    task = NullTask()
    task.add_hook(ViewerHook())
    env = QuadrupedEnv(rob, arena, task)
    env.reset()
    for _ in range(1000):
        env.step(rob.STANCE_CONFIG)


def test_tg():
    rob = Aliengo(500, 'actuator_net', True)
    arena = Plain()
    task = LocomotionPMTG()
    task.add_hook(ViewerHook())
    env = QuadrupedEnv(rob, arena, task)
    env.reset()
    for _ in range(1000):
        env.step(np.zeros(16))


def test_replaceHeightfield():
    pyb.connect(pyb.GUI)
    pyb.setRealTimeSimulation(True)
    terrains = [PlainHf.default(), Hills.default(),
                Steps.default(), Slopes.default()]
    current = None
    for i in range(5):
        for terrain in terrains:
            if current is None:
                terrain.spawn(pyb)
            else:
                terrain.replace(pyb, current)
            current = terrain
            time.sleep(0.5)
    current.remove(pyb)
    time.sleep(1.0)
    pyb.disconnect()


def test_terrainApi():
    pyb.connect(pyb.GUI)
    pyb.setRealTimeSimulation(True)
    terrain = Hills.make(20, 0.1, (0.5, 10), random_state=np.random)
    # terrain = Steps.make(20, 0.1, 1.0, 0.5, random_state=np.random)
    # terrain = Slopes.make(20, 0.1, np.pi / 6, 0.5)
    terrain.spawn(pyb)

    sphere_shape = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0., 0.8, 0., 0.6)
    )
    ray_hit_shape = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0.8, 0., 0., 0.6)
    )
    cylinder_shape = pyb.createVisualShape(
        shapeType=pyb.GEOM_CYLINDER, radius=0.005, length=0.11,
        rgbaColor=(0., 0, 0.8, 0.6)
    )
    box_shape = pyb.createVisualShape(
        shapeType=pyb.GEOM_BOX, halfExtents=(0.03, 0.03, 0.03),
        rgbaColor=(0.8, 0., 0., 0.6)
    )

    points, vectors, ray_hits = [], [], []
    box_id = -1

    for i in range(10):
        peak = terrain.get_peak((-1, 1), (-1, 1))
        if i == 0:
            box_id = pyb.createMultiBody(
                baseVisualShapeIndex=box_shape,
                basePosition=peak
            )
        else:
            pyb.resetBasePositionAndOrientation(
                box_id, peak, (0., 0., 0., 1.)
            )
        for idx, (x, y) in enumerate(
            itertools.product(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        ):
            h = terrain.get_height(x, y)
            vec_orn = tf.Quaternion.from_rotation(
                tf.Rotation.from_zaxis(terrain.get_normal(x, y))
            )
            ray_pos = pyb.rayTest((x, y, 2), (x, y, -1))[0][3]
            if i == 0:
                points.append(pyb.createMultiBody(
                    baseVisualShapeIndex=sphere_shape,
                    basePosition=(x, y, h)
                ))
                vectors.append(pyb.createMultiBody(
                    baseVisualShapeIndex=cylinder_shape,
                    basePosition=(x, y, h),
                    baseOrientation=vec_orn
                ))
                ray_hits.append(pyb.createMultiBody(
                    baseVisualShapeIndex=ray_hit_shape,
                    basePosition=ray_pos
                ))
            else:
                pyb.resetBasePositionAndOrientation(
                    points[idx], (x, y, h), (0., 0., 0., 1.)
                )
                pyb.resetBasePositionAndOrientation(
                    ray_hits[idx], ray_pos, (0., 0., 0., 1.)
                )
                pyb.resetBasePositionAndOrientation(
                    vectors[idx], (x, y, h), vec_orn
                )
        time.sleep(3)
        new = Hills.make(20, 0.1, (np.random.random(), 10), random_state=np.random)
        new.replace(pyb, terrain)
        terrain = new

    pyb.disconnect()


def test_gym_env():
    rob = Aliengo(500, 'pd')
    arena = Plain()
    task = LocomotionV0()
    env = QuadrupedEnv(rob, arena, task)
    print(env.observation_space)
    print(env.action_space)


if __name__ == '__main__':
    test_robot_extent()