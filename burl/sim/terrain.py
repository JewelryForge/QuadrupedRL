import math
import time
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import numpy as np
import pybullet as pyb
from scipy.interpolate import interp2d

from burl.utils import unit, vec_cross

__all__ = ['Terrain', 'Plain', 'HeightFieldTerrain', 'Steps', 'Slope', 'Stairs', 'Hills']


class Terrain(object):
    def __init__(self):
        self.terrain_id: int = -1

    @property
    def id(self):
        return self.terrain_id

    def spawn(self, sim_env):
        raise NotImplementedError

    def get_height(self, x, y):
        raise NotImplementedError

    def get_normal(self, x, y):
        raise NotImplementedError

    def get_peak(self, x_range: tuple[float, float], y_range: tuple[float, float]):
        raise NotImplementedError


class Plain(Terrain):
    def spawn(self, sim_env):
        self.terrain_id = sim_env.loadURDF("plane.urdf")
        sim_env.changeDynamics(self.terrain_id, -1, lateralFriction=1.0)

    def get_height(self, x, y):
        return 0.0

    def get_normal(self, x, y):
        return np.array((0, 0, 1))

    def get_peak(self, x_range, y_range):
        return sum(x_range) / 2, sum(y_range) / 2, 0.0


NUMERIC = Union[int, float]


@dataclass
class HeightField:
    data: np.ndarray
    size: Union[NUMERIC, tuple[NUMERIC]]
    resolution: NUMERIC


class HeightFieldTerrain(Terrain):
    def __init__(self, heightfield: HeightField):
        super().__init__()
        self.heightfield = heightfield.data
        self.offset = None
        self.y_dim, self.x_dim = self.heightfield.shape
        if isinstance(heightfield.size, Iterable):
            self.x_size = self.y_size = heightfield.size
        else:
            self.x_size, self.y_size = heightfield.size, heightfield.size
        self.x_rsl = self.y_rsl = heightfield.resolution
        self.terrain_shape_id = -1

    def spawn(self, sim_env, replace_id=-1, offset=(0., 0., 0.)):
        self.offset = np.asarray(offset)
        self.terrain_shape_id = sim_env.createCollisionShape(
            shapeType=pyb.GEOM_HEIGHTFIELD, flags=pyb.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=(self.x_rsl, self.y_rsl, 1.0),
            heightfieldTextureScaling=self.x_size,
            heightfieldData=self.heightfield.reshape(-1),
            numHeightfieldColumns=self.x_dim, numHeightfieldRows=self.y_dim,
            replaceHeightfieldIndex=replace_id)
        if replace_id == -1:
            self.terrain_id = sim_env.createMultiBody(0, self.terrain_shape_id)
            sim_env.changeVisualShape(self.terrain_id, -1, rgbaColor=(1, 1, 1, 1))
            sim_env.changeDynamics(self.terrain_id, -1, lateralFriction=1.0)
            origin_z = (np.max(self.heightfield) + np.min(self.heightfield)) / 2
            # does not need to move again when its height field is replaced
            sim_env.resetBasePositionAndOrientation(self.terrain_id, self.offset + (0, 0, origin_z),
                                                    (0., 0., 0., 1.))

    def replace_heightfield(self, sim_env, height_field: HeightField):
        self.heightfield = height_field.data
        self.spawn(sim_env, replace_id=self.terrain_shape_id)

    @property
    def shape_id(self):
        return self.terrain_shape_id

    def get_disc_x(self, x):
        return int((x + self.x_size / 2 - self.offset[0]) / self.x_rsl)

    def get_disc_y(self, y):
        return int((y + self.y_size / 2 - self.offset[1]) / self.y_rsl)

    def get_cont_x(self, x_idx):
        return x_idx * self.x_rsl - self.x_size / 2 + self.offset[0]

    def get_cont_y(self, y_idx):
        return y_idx * self.y_rsl - self.y_size / 2 + self.offset[1]

    def get_nearest_vertices(self, x, y):
        x_idx, y_idx = self.get_disc_x(x), self.get_disc_y(y)
        x_rnd, y_rnd = self.get_cont_x(x_idx), self.get_cont_y(y_idx)
        if (x - x_rnd) / self.x_rsl + (y - y_rnd) / self.y_rsl < 1:
            v1 = x_rnd, y_rnd, self.heightfield[y_idx, x_idx]
        else:
            v1 = x_rnd + self.x_rsl, y_rnd + self.y_rsl, self.heightfield[y_idx + 1, x_idx + 1]
        v2 = x_rnd, y_rnd + self.y_rsl, self.heightfield[y_idx + 1, x_idx]
        v3 = x_rnd + self.x_rsl, y_rnd, self.heightfield[y_idx, x_idx + 1]
        return np.array(v1), np.array(v2), np.array(v3)

    def get_peak(self, x_range, y_range):
        (x_lower, x_upper), (y_lower, y_upper) = x_range, y_range
        x_lower_idx, x_upper_idx = self.get_disc_x(x_lower), self.get_disc_x(x_upper) + 1
        y_lower_idx, y_upper_idx = self.get_disc_y(y_lower), self.get_disc_y(y_upper) + 1
        height_field_part = self.heightfield[y_lower_idx:y_upper_idx, x_lower_idx:x_upper_idx]
        y_size, x_size = height_field_part.shape
        max_idx = np.argmax(height_field_part)
        max_x_idx, max_y_idx = max_idx % x_size, max_idx // x_size
        max_height = height_field_part[max_y_idx, max_x_idx]
        max_x, max_y = x_lower + max_x_idx * self.x_rsl, y_lower + max_y_idx * self.y_rsl
        return max_x, max_y, max_height

    def get_height(self, x, y):
        try:
            v1, v2, v3 = self.get_nearest_vertices(x, y)
        except IndexError:
            return 0.0
        if x == v1[0] and y == v1[1]:
            return v1[2]
        x1, y1, z1 = v2 - v1
        x2, y2, z2 = v3 - v1
        x3, y3 = x - v1[0], y - v1[1]
        div = (x1 * y2 - x2 * y1)
        c1 = (x3 * y2 - x2 * y3) / div
        c2 = (x1 * y3 - x3 * y1) / div
        return c1 * z1 + c2 * z2 + v1[2] + self.offset[2]

    def get_normal(self, x, y) -> np.ndarray:
        try:
            v1, v2, v3 = self.get_nearest_vertices(x, y)
        except IndexError:
            return np.array((0., 0., 1.))
        normal = unit(vec_cross(v1 - v2, v1 - v3))
        return normal if normal[2] > 0 else -normal


class Steps(HeightFieldTerrain):
    @classmethod
    def make(cls, size, resolution, step_width, max_step_height):
        return cls(cls.make_heightfield(size, resolution, step_width, max_step_height))

    @staticmethod
    def make_heightfield(size, resolution, step_width, max_step_height):
        step = int(step_width / resolution)
        data_size = int(size / resolution) + 1
        num_steps = int(data_size / step) + 1
        height_field_data = np.zeros((data_size, data_size))
        for i in range(num_steps):
            for j in range(num_steps):
                x_start, x_stop, y_start, y_stop = i * step, (i + 1) * step, j * step, (j + 1) * step
                height_field_data[y_start:y_stop, x_start:x_stop] = random.uniform(0., max_step_height)
        return HeightField(height_field_data, size, resolution)


class Slope(HeightFieldTerrain):
    @classmethod
    def make(cls, size, resolution, slope, slope_width, axis='x'):
        return cls(cls.make_heightfield(size, resolution, slope, slope_width, axis))

    @classmethod
    def make_heightfield(cls, size, resolution, slope, slope_width, axis):
        step = int(slope_width * 2 / resolution)
        data_size = int(size / resolution) + 1
        num_steps = int(data_size / step) + 1
        slope = math.tan(slope)
        height_field_data = np.zeros((data_size, data_size))
        for i in range(num_steps):
            x_start, x_stop = i * step, int((i + 0.5) * step)
            for j in range(x_start, min(x_stop, data_size)):
                height_field_data[:, j] = (j - x_start) * slope * resolution
            x_start, x_stop = x_stop, (i + 1) * step
            for j in range(x_start, min(x_stop, data_size)):
                height_field_data[:, j] = (x_stop - j) * slope * resolution
        return HeightField(cls.rotate(height_field_data, axis), size, resolution)

    @classmethod
    def rotate(cls, height_field_data, axis):
        if axis == 'x':
            return height_field_data
        elif axis == 'y':
            return height_field_data.T
        raise RuntimeError('Unknown axis')


class Stairs(HeightFieldTerrain):
    @classmethod
    def make(cls, size, resolution, slope, slope_width, axis='+x'):
        return cls(cls.make_heightfield(size, resolution, slope, slope_width, axis))

    @classmethod
    def make_heightfield(cls, size, resolution, stair_height, stair_width, axis):
        step = int(stair_width * 2 / resolution)
        data_size = int(size / resolution) + 1
        num_steps = int(data_size / step) + 1
        height_field_data = np.zeros((data_size, data_size))
        height = -stair_height * int(num_steps / 2)
        for i in range(num_steps):
            x_start, x_stop = i * step, (i + 1) * step
            height_field_data[:, x_start:x_stop] = height
            height += stair_height
        return HeightField(cls.rotate(height_field_data, axis), size, resolution)

    @classmethod
    def rotate(cls, height_field_data, axis):
        if axis == '+x':
            return height_field_data
        elif axis == '+y':
            return height_field_data.T
        elif axis == '-x':
            return np.fliplr(height_field_data)
        elif axis == '-y':
            return np.flipud(height_field_data.T)
        raise RuntimeError('Unknown axis')


class Hills(HeightFieldTerrain):
    def __init__(self, heightfield, seed=None):
        np.random.seed(seed)
        super().__init__(heightfield)

    @classmethod
    def make(cls, size, resolution, *roughness_downsample: tuple[NUMERIC, NUMERIC]):
        return cls(cls.make_heightfield(size, resolution, *roughness_downsample))

    @classmethod
    def make_heightfield(cls, size, resolution, *roughness_downsample: tuple[NUMERIC, NUMERIC]) -> HeightField:
        height_field_data = None
        for roughness, downsample in roughness_downsample:
            sample_rsl = downsample * resolution
            x = y = np.arange(-size / 2 - 3 * sample_rsl, size / 2 + 4 * sample_rsl, sample_rsl)
            height_field_downsampled = np.random.uniform(0, roughness, (x.size, y.size))
            terrain_func = interp2d(x, y, height_field_downsampled, kind='cubic')

            data_size = int(size / resolution) + 1
            x_upsampled = y_upsampled = np.linspace(-size / 2, size / 2, data_size)
            if height_field_data is None:
                height_field_data = terrain_func(x_upsampled, y_upsampled)
            else:
                height_field_data += terrain_func(x_upsampled, y_upsampled)
        return HeightField(height_field_data, size, resolution)

    # def get_height(self, x, y):
    #     return self.terrain_func(x, y).squeeze() + self.offset[2]
    #
    # def get_nearest_vertices(self, x, y):
    #     res = super().getNearestVertices(x, y)
    #     residue = np.array([z - self.getHeight(x, y) for x, y, z in res])
    #     if any(residue > 1e-5):
    #         print(residue)
    #     return res


if __name__ == '__main__':
    from burl.sim.env import AlienGo

    pyb.connect(pyb.GUI)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    t = Stairs.make_heightfield(20, 0.05, 0.15, 0.3, '+y')
    # t.data += Hills.make_heightfield(20, 0.05, (0.02, 1)).data
    t = Stairs(t)
    t.spawn(pyb)
    # robot = AlienGo()
    # robot.spawn()

    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    pyb.setGravity(0, 0, -10)
    pyb.setRealTimeSimulation(1)

    pyb.changeVisualShape(t.id, -1, rgbaColor=(1, 1, 1, 1))

    # terrain_visual_shape = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0., 0.8, 0., 0.6))
    # ray_hit_shape = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0.8, 0., 0., 0.6))
    # cylinder_shape = pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=0.005, length=0.11,
    #                                        rgbaColor=(0., 0, 0.8, 0.6))
    # box_shape = pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=(0.03, 0.03, 0.03),
    #                                   rgbaColor=(0.8, 0., 0., 0.6))
    #
    # from burl.utils.transforms import Quaternion
    #
    # points, vectors, ray_hits = [], [], []
    #
    # for _x in np.linspace(-1, 1, 10):
    #     for _y in np.linspace(-1, 1, 10):
    #         h = t.get_height(_x, _y)
    #         points.append(pyb.createMultiBody(baseVisualShapeIndex=terrain_visual_shape,
    #                                           basePosition=(_x, _y, h)))
    #         n = t.get_normal(_x, _y)
    #         y_ax = unit(np.cross(n, (1, 0, 0)))
    #         x_ax = unit(np.cross(y_ax, n))
    #         vectors.append(pyb.createMultiBody(
    #             baseVisualShapeIndex=cylinder_shape, basePosition=(_x, _y, h),
    #             baseOrientation=Quaternion.from_rotation(np.array((x_ax, y_ax, n)).T)))
    #         ray_hits.append(pyb.createMultiBody(
    #             baseVisualShapeIndex=ray_hit_shape, basePosition=pyb.rayTest((_x, _y, 2), (_x, _y, -1))[0][3]
    #         ))
    #
    # cor = t.get_peak((-0.5, 0.5), (-0.5, 0.5))
    # box_id = pyb.createMultiBody(baseVisualShapeIndex=box_shape,
    #                              basePosition=cor, baseOrientation=(0., 0., 0., 1.))
    #
    # for i in range(10):
    #     time.sleep(5)
    #     pyb.resetBasePositionAndOrientation(robot.id, (0., 0., 3.), (0., 0., 0., 1.))
    #     t.replace_heightfield(pyb, Hills.make_heightfield(3, 0.1, (random.random(), 20)))
    #     idx = 0
    #     for _x in np.linspace(-1, 1, 10):
    #         for _y in np.linspace(-1, 1, 10):
    #             h = t.get_height(_x, _y)
    #             pyb.resetBasePositionAndOrientation(points[idx], (_x, _y, h), (0., 0., 0., 1.))
    #             pyb.resetBasePositionAndOrientation(ray_hits[idx], pyb.rayTest((_x, _y, 2), (_x, _y, -1))[0][3],
    #                                                 (0., 0., 0., 1.))
    #             n = t.get_normal(_x, _y)
    #             y_ax = unit(np.cross(n, (1, 0, 0)))
    #             x_ax = unit(np.cross(y_ax, n))
    #             pyb.resetBasePositionAndOrientation(vectors[idx], (_x, _y, h),
    #                                                 Quaternion.from_rotation(np.array((x_ax, y_ax, n)).T))
    #             idx += 1
    #     pyb.resetBasePositionAndOrientation(box_id, t.get_peak((-0.5, 0.5), (-0.5, 0.5)), (0., 0., 0., 1.))
    time.sleep(300)
