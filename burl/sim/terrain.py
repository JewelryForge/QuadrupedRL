import time

import pybullet
import numpy as np
from scipy.interpolate import interp2d

from burl.utils import unit


class Terrain(object):
    def __init__(self):
        self.terrain_id = None
        pass

    @property
    def id(self):
        return self.terrain_id

    def reset(self):
        pass

    def getHeight(self, x, y):
        raise NotImplementedError

    def getNormal(self, x, y):
        raise NotImplementedError


class PlainTerrain(Terrain):
    def __init__(self, bullet_client):
        super().__init__()
        self.terrain_id = bullet_client.loadURDF("plane.urdf")
        bullet_client.changeDynamics(self.terrain_id, -1, lateralFriction=5.0)

    def getHeight(self, x, y):
        return 0

    def getNormal(self, x, y):
        return np.array((0, 0, 1))


class RandomUniformTerrain(Terrain):
    def __init__(self,
                 bullet_client,
                 size=15,
                 downsample=10,
                 roughness=0.1,
                 resolution=0.02,
                 offset=(0, 0, 0),
                 seed=None):
        super().__init__()
        self.size, self.offset = size, np.asarray(offset, dtype=float)
        sample_resolution = downsample * resolution
        np.random.seed(seed)
        x = np.arange(-size / 2 - 3 * sample_resolution, size / 2 + 4 * sample_resolution, sample_resolution)
        y = x.copy()
        height_field_downsampled = np.random.uniform(0, roughness, (x.size, y.size))
        self.terrain_func = interp2d(x, y, height_field_downsampled, kind='cubic')

        self.resolution = resolution
        data_size = int(size / resolution) + 1
        x_upsampled = np.linspace(-size / 2, size / 2, data_size)
        y_upsampled = np.linspace(-size / 2, size / 2, data_size)
        self.height_field = self.terrain_func(x_upsampled, y_upsampled)
        terrain_shape = bullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=(resolution, resolution, 1.0),
            heightfieldTextureScaling=size,
            heightfieldData=self.height_field.reshape(-1),
            numHeightfieldColumns=data_size, numHeightfieldRows=data_size)
        self.terrain_id = bullet_client.createMultiBody(0, terrain_shape, -1,
                                                        self.offset + (0, 0, np.mean(self.height_field).item()),
                                                        (0, 0, 0, 1))
        bullet_client.changeVisualShape(self.terrain_id, -1, rgbaColor=(1, 1, 1, 1))
        bullet_client.changeDynamics(self.terrain_id, -1, lateralFriction=5.0)

    def getNearestVertices(self, x, y):
        x_ref = x + self.size / 2 - self.offset[0]
        y_ref = y + self.size / 2 - self.offset[1]
        r = self.resolution
        x_idx, y_idx = int(x_ref // r), int(y_ref // r)
        x_rnd, y_rnd = x - x % r, y - y % r,
        if x % r + y % r < r:
            v1 = x_rnd, y_rnd, self.height_field[y_idx, x_idx]
        else:
            v1 = x_rnd + r, y_rnd + r, self.height_field[y_idx + 1, x_idx + 1]
        v2 = x_rnd, y_rnd + r, self.height_field[y_idx + 1, x_idx]
        v3 = x_rnd + r, y_rnd, self.height_field[y_idx, x_idx + 1]
        return np.array(v1), np.array(v2), np.array(v3)

    def getHeight(self, x, y):
        return self.terrain_func(x, y).squeeze() + self.offset[2]

    def getNormal(self, x, y):
        try:
            v1, v2, v3 = self.getNearestVertices(x, y)
        except IndexError:
            return np.array((0, 0, 1))
        normal = unit(np.cross(v1 - v2, v1 - v3))
        return normal if normal[2] > 0 else -normal


if __name__ == '__main__':
    pybullet.connect(pybullet.GUI)
    t = RandomUniformTerrain(pybullet, size=2)
    pybullet.changeVisualShape(t.id, -1, rgbaColor=(1, 1, 1, 1))
    terrain_visual_shape = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                      radius=0.01,
                                                      rgbaColor=(0., 0.8, 0., 0.6))
    cylinder_shape = pybullet.createVisualShape(shapeType=pybullet.GEOM_CYLINDER,
                                                radius=0.005, length=0.11,
                                                rgbaColor=(0., 0, 0.8, 0.6))

    from burl.utils.transforms import Quaternion

    for x in np.linspace(-1, 1, 11):
        for y in np.linspace(-1, 1, 11):
            h = t.getHeight(x, y)
            pybullet.createMultiBody(baseVisualShapeIndex=terrain_visual_shape,
                                     basePosition=(x, y, h))
            n = t.getNormal(x, y)
            y_ax = unit(np.cross(n, (1, 0, 0)))
            x_ax = unit(np.cross(y_ax, n))
            pybullet.createMultiBody(baseVisualShapeIndex=cylinder_shape,
                                     basePosition=(x, y, h),
                                     baseOrientation=(Quaternion.from_rotation(np.array((x_ax, y_ax, n)).T)))
    for _ in range(100000):
        pybullet.stepSimulation()
        time.sleep(1 / 240)
