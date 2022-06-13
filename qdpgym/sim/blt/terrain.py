import abc
import dataclasses
from typing import Union, Tuple, Iterable

import numpy as np
import pybullet as pyb
from scipy.interpolate import interp2d

from qdpgym.sim.abc import Terrain, NUMERIC
from qdpgym.utils.tf import vcross, vunit

__all__ = [
    'NullTerrain', 'Plain', 'HeightFieldTerrain',
    'Hills', 'PlainHf', 'Slopes', 'Steps', 'TerrainBase'
]


class TerrainBase(Terrain, metaclass=abc.ABCMeta):
    def __init__(self):
        self._id: int = -1
        self._spawned = False

    @property
    def id(self):
        return self._id

    def spawn(self, sim_env):
        raise NotImplementedError

    def replace(self, sim_env, obj: 'TerrainBase'):
        obj.remove(sim_env)
        self.spawn(sim_env)

    def remove(self, sim_env):
        if self._id != -1:
            sim_env.removeBody(self._id)


class NullTerrain(TerrainBase):
    pass


class Plain(TerrainBase):
    def spawn(self, sim_env):
        if self._id != -1:
            return
        self._id = sim_env.loadURDF("plane.urdf")
        sim_env.changeDynamics(self._id, -1, lateralFriction=1.0)

    def get_height(self, x, y):
        return 0.0

    def get_normal(self, x, y):
        return np.array((0, 0, 1))

    def get_peak(self, x_range, y_range):
        return sum(x_range) / 2, sum(y_range) / 2, 0.0

    def out_of_range(self, x, y):
        return False


@dataclasses.dataclass
class HeightField:
    data: np.ndarray
    size: Union[NUMERIC, Tuple[NUMERIC]]
    resolution: NUMERIC


class HeightFieldTerrain(TerrainBase):
    def __init__(self, heightfield: HeightField):
        super().__init__()
        self.heightfield = heightfield.data
        self.y_dim, self.x_dim = self.heightfield.shape
        if isinstance(heightfield.size, Iterable):
            self.x_size, self.y_size = heightfield.size
        else:
            self.x_size, self.y_size = heightfield.size, heightfield.size
        self.x_rsl = self.y_rsl = heightfield.resolution
        self._shape_id = -1

    def spawn(self, sim_env):
        if self._id != -1:
            return
        self._shape_id = sim_env.createCollisionShape(
            shapeType=pyb.GEOM_HEIGHTFIELD,
            flags=pyb.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=(self.x_rsl, self.y_rsl, 1.0),
            heightfieldTextureScaling=self.x_size,
            heightfieldData=self.heightfield.reshape(-1),
            numHeightfieldColumns=self.x_dim,
            numHeightfieldRows=self.y_dim,
            replaceHeightfieldIndex=-1
        )
        self._id = sim_env.createMultiBody(0, self._shape_id)
        sim_env.changeVisualShape(self._id, -1, rgbaColor=(1, 1, 1, 1))
        sim_env.changeDynamics(self._id, -1, lateralFriction=1.0)
        origin_z = (np.max(self.heightfield) + np.min(self.heightfield)) / 2
        sim_env.resetBasePositionAndOrientation(self._id, (0, 0, origin_z),
                                                (0., 0., 0., 1.))

    def replace(self, sim_env, obj):
        assert self._id == self._shape_id == -1, f'`{self}` have been spawned'
        if obj._id == -1:
            return

        if not isinstance(obj, HeightFieldTerrain):
            return super().replace(sim_env, obj)

        # Currently, in bullet <= 3.2.4,
        # Heightfield replacement may cause collision detection failure.
        # See https://github.com/bulletphysics/bullet3/issues/4236
        # See https://github.com/bulletphysics/bullet3/pull/4253
        self._id = obj._id
        self._shape_id = sim_env.createCollisionShape(
            shapeType=pyb.GEOM_HEIGHTFIELD, flags=pyb.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=(self.x_rsl, self.y_rsl, 1.0),
            heightfieldTextureScaling=self.x_size,
            heightfieldData=self.heightfield.reshape(-1),
            numHeightfieldColumns=self.x_dim, numHeightfieldRows=self.y_dim,
            replaceHeightfieldIndex=obj._shape_id
        )
        obj._id = obj._shape_id = -1

        origin_z = (np.max(self.heightfield) + np.min(self.heightfield)) / 2
        sim_env.resetBasePositionAndOrientation(self._id, (0, 0, origin_z),
                                                (0., 0., 0., 1.))

    def remove(self, sim_env):
        if self._id != -1:
            sim_env.removeBody(self._id)
            sim_env.removeCollisionShape(self._shape_id)
            self._id = self._shape_id = -1

    @property
    def shape_id(self):
        return self._shape_id

    def get_disc_x(self, x):
        return int((x + self.x_size / 2) / self.x_rsl)

    def get_disc_y(self, y):
        return int((y + self.y_size / 2) / self.y_rsl)

    def get_cont_x(self, x_idx):
        return x_idx * self.x_rsl - self.x_size / 2

    def get_cont_y(self, y_idx):
        return y_idx * self.y_rsl - self.y_size / 2

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
        hfield_part = self.heightfield[y_lower_idx:y_upper_idx, x_lower_idx:x_upper_idx]
        y_size, x_size = hfield_part.shape
        max_idx = np.argmax(hfield_part)
        max_x_idx, max_y_idx = max_idx % x_size, max_idx // x_size
        max_height = hfield_part[max_y_idx, max_x_idx]
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
        return c1 * z1 + c2 * z2 + v1[2]

    def get_normal(self, x, y) -> np.ndarray:
        try:
            v1, v2, v3 = self.get_nearest_vertices(x, y)
        except IndexError:
            return np.array((0., 0., 1.))
        normal = vunit(vcross(v1 - v2, v1 - v3))
        return normal if normal[2] > 0 else -normal

    def out_of_range(self, x, y):
        return abs(x) > self.x_size / 2 - 1 or abs(y) > self.y_size / 2 - 1


class Hills(HeightFieldTerrain):
    def __init__(self, heightfield):
        super().__init__(heightfield)

    @classmethod
    def make(cls, size, resolution, *rough_dsp: Tuple[NUMERIC, NUMERIC], random_state):
        return cls(cls.make_hfield(size, resolution, *rough_dsp, random_state=random_state))

    @classmethod
    def make_hfield(cls, size, resol, *rough_dsp: Tuple[NUMERIC, NUMERIC], random_state) -> HeightField:
        data_size = int(size / resol) + 1
        hfield_data = np.zeros((data_size, data_size))
        for roughness, dsp in rough_dsp:
            sample_rsl = dsp * resol
            x = y = np.arange(-size / 2 - 3 * sample_rsl, size / 2 + 4 * sample_rsl, sample_rsl)
            hfield_dsp = random_state.uniform(0, roughness, (x.size, y.size))
            terrain_func = interp2d(x, y, hfield_dsp, kind='cubic')
            x_upsampled = y_upsampled = np.linspace(-size / 2, size / 2, data_size)
            hfield_data += terrain_func(x_upsampled, y_upsampled)
        return HeightField(hfield_data, size, resol)

    @classmethod
    def default(cls):
        return cls.make(20, 0.1, (0.2, 10), random_state=np.random)


class PlainHf(HeightFieldTerrain):
    @classmethod
    def make(cls, size, resolution):
        return cls(cls.make_hfield(size, resolution))

    @staticmethod
    def make_hfield(size, resolution):
        data_size = int(size / resolution) + 1
        hfield_data = np.zeros((data_size, data_size))
        return HeightField(hfield_data, size, resolution)

    def get_height(self, x, y):
        return 0.0

    def get_normal(self, x, y):
        return np.array((0., 0., 1.))

    def get_peak(self, x_range, y_range):
        return sum(x_range) / 2, sum(y_range) / 2, 0.

    @classmethod
    def default(cls):
        return cls.make(20, 0.1)


class Slopes(HeightFieldTerrain):
    @classmethod
    def make(cls, size, resolution, slope, slope_width, axis='x'):
        return cls(cls.make_hfield(size, resolution, slope, slope_width, axis))

    @classmethod
    def make_hfield(cls, size, resolution, slope, slope_width, axis):
        step = int(slope_width * 2 / resolution)
        data_size = int(size / resolution) + 1
        num_steps = int(data_size / step) + 1
        slope = np.tan(slope)
        hfield_data = np.zeros((data_size, data_size))
        for i in range(num_steps):
            x_start, x_stop = i * step, int((i + 0.4) * step)
            for j in range(x_start, min(x_stop, data_size)):
                hfield_data[:, j] = (x_stop - j) * slope * resolution
            x_start, x_stop = x_stop, int((i + 0.6) * step)
            for j in range(x_start, min(x_stop, data_size)):
                hfield_data[:, j] = 0.
            x_start, x_stop = x_stop, (i + 1) * step
            for j in range(x_start, min(x_stop, data_size)):
                hfield_data[:, j] = (j - x_start) * slope * resolution
        return HeightField(cls.rotate(hfield_data, axis), size, resolution)

    @classmethod
    def rotate(cls, hfield_data, axis):
        if axis == 'x':
            return hfield_data
        elif axis == 'y':
            return hfield_data.T
        raise RuntimeError('Unknown axis')

    @classmethod
    def default(cls):
        return cls.make(20, 0.1, 0.17, 3.0)


class Steps(HeightFieldTerrain):
    @classmethod
    def make(cls, size, resolution, step_width, max_step_height, random_state):
        return cls(cls.make_hfield(size, resolution, step_width, max_step_height, random_state))

    @staticmethod
    def make_hfield(size, resolution, step_width, max_step_height, random_state):
        step = int(step_width / resolution)
        data_size = int(size / resolution) + 1
        num_steps = int(data_size / step) + 1
        hfield_data = np.zeros((data_size, data_size))
        for i in range(num_steps + 1):
            for j in range(num_steps + 1):
                x_start, x_stop = int((i - 0.5) * step), int((i + 0.5) * step)
                y_start, y_stop = int((j - 0.5) * step), int((j + 0.5) * step)
                hfield_data[y_start:y_stop, x_start:x_stop] = random_state.uniform(0., max_step_height)
        return HeightField(hfield_data, size, resolution)

    @classmethod
    def default(cls):
        return cls.make(20, 0.1, 1.0, 0.1, random_state=np.random)
