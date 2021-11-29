import time

import pybullet
import numpy as np
from scipy.interpolate import interp2d


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
        raise np.array((0, 0, 1))


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
        self.size, self.offset = size, offset
        data_size = np.asarray((size, size)) / resolution
        x_size, y_size = data_size.astype(int)
        sample_size = (data_size / downsample).astype(int)
        np.random.seed(seed)
        height_field_downsampled = np.random.uniform(0, roughness, sample_size)
        x = np.arange(0, x_size, downsample)
        y = np.arange(0, y_size, downsample)

        self.terrain_func = interp2d(x, y, height_field_downsampled, kind='cubic')
        self.resolution = resolution
        x_upsampled = np.arange(x_size)
        y_upsampled = np.arange(y_size)
        z_upsampled = self.terrain_func(x_upsampled, y_upsampled)
        terrain_shape = bullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD, meshScale=(resolution, resolution, 1.0),
            heightfieldTextureScaling=size / 2,
            heightfieldData=z_upsampled.reshape(-1), numHeightfieldColumns=x_size, numHeightfieldRows=y_size)

        self.terrain_id = bullet_client.createMultiBody(0, terrain_shape)
        bullet_client.resetBasePositionAndOrientation(self.terrain_id, offset, (0, 0, 0, 1))
        bullet_client.changeDynamics(self.terrain_id, -1, lateralFriction=5.0)

    def getHeight(self, x, y):
        x = (x + self.size / 2 - self.offset[0]) / self.resolution
        y = (y + self.size / 2 - self.offset[1]) / self.resolution
        return self.terrain_func(x, y).squeeze() + self.offset[2]

    def getNormal(self, x, y):
        x = x - self.size / 2 - self.offset[0]
        y = y - self.size / 2 - self.offset[1]

        # x_idx, y_idx = x // self.resolution, y // self.resolution
        #
        #
        # na = (v2.y - v1.y) * (v3.z - v1.z) - (v2.z - v1.z) * (v3.y - v1.y);
        # double
        # nb = (v2.z - v1.z) * (v3.x - v1.x) - (v2.x - v1.x) * (v3.z - v1.z);
        # double
        # nc = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);
        raise NotImplementedError


if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000)
    pybullet.connect(pybullet.GUI)
    t = RandomUniformTerrain(pybullet)
    print(t.getHeight(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)), end=' ')
    for _ in range(100000):
        pybullet.stepSimulation()
        time.sleep(1 / 240)
