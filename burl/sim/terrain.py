import time
from collections.abc import Iterable

import numpy as np
import pybullet as pyb
from scipy.interpolate import interp2d

from burl.utils import unit, vec_cross


class Terrain(object):
    def __init__(self):
        self.terrain_id: int = -1

    @property
    def id(self):
        return self.terrain_id

    def spawn(self, sim_env):
        raise NotImplementedError

    def getHeight(self, x, y):
        raise NotImplementedError

    def getNormal(self, x, y):
        raise NotImplementedError

    def getPeakInRegion(self, x_range, y_range):
        raise NotImplementedError


class Plain(Terrain):
    def spawn(self, sim_env):
        self.terrain_id = sim_env.loadURDF("plane.urdf")
        sim_env.changeDynamics(self.terrain_id, -1, lateralFriction=1.0)

    def getHeight(self, x, y):
        return 0.0

    def getNormal(self, x, y):
        return np.array((0, 0, 1))

    def getPeakInRegion(self, x_range, y_range):
        return sum(x_range) / 2, sum(y_range) / 2, 0.0


class HeightFieldTerrain(Terrain):
    def __init__(self, height_field, resolution, offset=(0., 0., 0.)):
        super().__init__()
        self.height_field = np.asarray(height_field)
        self.offset = np.asarray(offset, dtype=float)
        self.y_dim, self.x_dim = self.height_field.shape
        self.x_size, self.y_size = (self.x_dim - 1) * resolution, (self.y_dim - 1) * resolution
        if isinstance(resolution, Iterable):
            try:
                self.x_rsl, self.y_rsl, self.z_rsl = resolution
            except ValueError:
                self.x_rsl, self.y_rsl = resolution
                self.z_rsl = 1.0
        else:
            self.x_rsl = self.y_rsl = resolution
            self.z_rsl = 1.0
        self.terrain_shape_id = -1

    def spawn(self, sim_env, replace_id=-1):
        self.terrain_shape_id = sim_env.createCollisionShape(
            shapeType=pyb.GEOM_HEIGHTFIELD, flags=pyb.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=(self.x_rsl, self.y_rsl, self.z_rsl),
            heightfieldTextureScaling=self.x_size,
            heightfieldData=self.height_field.reshape(-1),
            numHeightfieldColumns=self.x_dim, numHeightfieldRows=self.y_dim,
            replaceHeightfieldIndex=replace_id)
        if replace_id == -1:
            self.terrain_id = sim_env.createMultiBody(0, self.terrain_shape_id)
            sim_env.changeVisualShape(self.terrain_id, -1, rgbaColor=(1, 1, 1, 1))
            sim_env.changeDynamics(self.terrain_id, -1, lateralFriction=1.0)
            origin_z = (np.max(self.height_field) + np.min(self.height_field)) / 2
            # does not need to move again when its height field is replaced
            sim_env.resetBasePositionAndOrientation(self.terrain_id, self.offset + (0, 0, origin_z),
                                                    (0., 0., 0., 1.))

    def replaceHeightField(self, sim_env, height_field):
        self.height_field = height_field
        self.spawn(sim_env, replace_id=self.terrain_shape_id)

    @property
    def shape_id(self):
        return self.terrain_shape_id

    def xCoord2Idx(self, x):
        return int((x + 1e-10 + self.x_size / 2 - self.offset[0]) / self.x_rsl)

    def yCoord2Idx(self, y):
        return int((y + 1e-10 + self.y_size / 2 - self.offset[1]) / self.y_rsl)

    def xIdx2Coord(self, x_idx):
        return x_idx * self.x_rsl - self.x_size / 2 + self.offset[0]

    def yIdx2Coord(self, y_idx):
        return y_idx * self.y_rsl - self.y_size / 2 + self.offset[1]

    def getNearestVertices(self, x, y):
        x_idx, y_idx = self.xCoord2Idx(x), self.yCoord2Idx(y)
        x_rnd, y_rnd = self.xIdx2Coord(x_idx), self.yIdx2Coord(y_idx)
        if (x - x_rnd) / self.x_rsl + (y - y_rnd) / self.y_rsl < 1:
            v1 = x_rnd, y_rnd, self.height_field[y_idx, x_idx]
        else:
            v1 = x_rnd + self.x_rsl, y_rnd + self.y_rsl, self.height_field[y_idx + 1, x_idx + 1]
        v2 = x_rnd, y_rnd + self.y_rsl, self.height_field[y_idx + 1, x_idx]
        v3 = x_rnd + self.x_rsl, y_rnd, self.height_field[y_idx, x_idx + 1]
        return np.array(v1), np.array(v2), np.array(v3)

    def getPeakInRegion(self, x_range, y_range):
        (x_lower, x_upper), (y_lower, y_upper) = x_range, y_range
        x_lower_idx, x_upper_idx = self.xCoord2Idx(x_lower), self.xCoord2Idx(x_upper) + 1
        y_lower_idx, y_upper_idx = self.yCoord2Idx(y_lower), self.yCoord2Idx(y_upper) + 1
        height_field_part = self.height_field[y_lower_idx:y_upper_idx, x_lower_idx:x_upper_idx]
        y_size, x_size = height_field_part.shape
        max_idx = np.argmax(height_field_part)
        max_x_idx, max_y_idx = max_idx % x_size, max_idx // x_size
        max_height = height_field_part[max_y_idx, max_x_idx]
        max_x, max_y = x_lower + max_x_idx * self.x_rsl, y_lower + max_y_idx * self.y_rsl
        return max_x, max_y, max_height

    def getHeight(self, x, y):
        try:
            v1, v2, v3 = self.getNearestVertices(x, y)
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

    def getNormal(self, x, y) -> np.ndarray:
        try:
            v1, v2, v3 = self.getNearestVertices(x, y)
        except IndexError:
            return np.array((0., 0., 1.))
        normal = unit(vec_cross(v1 - v2, v1 - v3))
        return normal if normal[2] > 0 else -normal


class Steps(HeightFieldTerrain):
    pass


class Slope(HeightFieldTerrain):
    def __init__(self, slope, size, resolution, offset=(0., 0., 0.)):
        data_size = int(size / resolution) + 1
        x = np.linspace(-size / 2, size / 2, data_size)
        y = x.copy()
        height_field = np.tile(x * np.tan(slope), (len(y), 1))
        super().__init__(height_field, resolution, offset)


class Hills(HeightFieldTerrain):
    def __init__(self, size, downsample, roughness, resolution, offset=(0., 0., 0.), seed=None):
        height_field = self.makeHeightField(size, downsample, roughness, resolution, seed)
        super().__init__(height_field, resolution, offset)

    @staticmethod
    def makeHeightField(size, downsample, roughness, resolution, seed=None):
        np.random.seed(seed)
        sample_rsl = downsample * resolution
        x = np.arange(-size / 2 - 3 * sample_rsl, size / 2 + 4 * sample_rsl, sample_rsl)
        y = x.copy()
        height_field_downsampled = np.random.uniform(0, roughness, (x.size, y.size))
        terrain_func = interp2d(x, y, height_field_downsampled, kind='cubic')

        data_size = int(size / resolution) + 1
        x_upsampled = np.linspace(-size / 2, size / 2, data_size)
        y_upsampled = x_upsampled.copy()
        height_field = terrain_func(x_upsampled, y_upsampled)
        return height_field

    # def getHeight(self, x, y):
    #     return self.terrain_func(x, y).squeeze() + self.offset[2]
    #
    # def getNearestVertices(self, x, y):
    #     res = super().getNearestVertices(x, y)
    #     residue = np.array([z - self.getHeight(x, y) for x, y, z in res])
    #     if any(residue > 1e-5):
    #         print(residue)
    #     return res


if __name__ == '__main__':
    import random
    from burl.sim import AlienGo

    pyb.connect(pyb.GUI)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    t = Hills(size=3, roughness=0.4, downsample=20, resolution=0.1)
    t.spawn(pyb)
    robot = AlienGo()
    robot.spawn()

    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    pyb.setGravity(0, 0, -10)
    pyb.setRealTimeSimulation(1)

    pyb.changeVisualShape(t.id, -1, rgbaColor=(1, 1, 1, 1))

    terrain_visual_shape = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0., 0.8, 0., 0.6))
    ray_hit_shape = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.01, rgbaColor=(0.8, 0., 0., 0.6))
    cylinder_shape = pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=0.005, length=0.11,
                                           rgbaColor=(0., 0, 0.8, 0.6))
    box_shape = pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=(0.03, 0.03, 0.03),
                                      rgbaColor=(0.8, 0., 0., 0.6))

    from burl.utils.transforms import Quaternion

    points, vectors, ray_hits = [], [], []

    for x in np.linspace(-1, 1, 10):
        for y in np.linspace(-1, 1, 10):
            h = t.getHeight(x, y)
            points.append(pyb.createMultiBody(baseVisualShapeIndex=terrain_visual_shape,
                                              basePosition=(x, y, h)))
            n = t.getNormal(x, y)
            y_ax = unit(np.cross(n, (1, 0, 0)))
            x_ax = unit(np.cross(y_ax, n))
            vectors.append(pyb.createMultiBody(
                baseVisualShapeIndex=cylinder_shape, basePosition=(x, y, h),
                baseOrientation=Quaternion.from_rotation(np.array((x_ax, y_ax, n)).T)))
            ray_hits.append(pyb.createMultiBody(
                baseVisualShapeIndex=ray_hit_shape, basePosition=pyb.rayTest((x, y, 2), (x, y, -1))[0][3]
            ))

    cor = t.getPeakInRegion((-0.5, 0.5), (-0.5, 0.5))
    box_id = pyb.createMultiBody(baseVisualShapeIndex=box_shape,
                                 basePosition=cor, baseOrientation=(0., 0., 0., 1.))

    for i in range(10):
        time.sleep(5)
        pyb.resetBasePositionAndOrientation(robot.id, (0., 0., 3.), (0., 0., 0., 1.))
        t.replaceHeightField(pyb, Hills(size=3, roughness=random.random(), downsample=20,
                                        resolution=0.1).height_field)
        idx = 0
        for x in np.linspace(-1, 1, 10):
            for y in np.linspace(-1, 1, 10):
                h = t.getHeight(x, y)
                pyb.resetBasePositionAndOrientation(points[idx], (x, y, h), (0., 0., 0., 1.))
                pyb.resetBasePositionAndOrientation(ray_hits[idx], pyb.rayTest((x, y, 2), (x, y, -1))[0][3],
                                                    (0., 0., 0., 1.))
                n = t.getNormal(x, y)
                y_ax = unit(np.cross(n, (1, 0, 0)))
                x_ax = unit(np.cross(y_ax, n))
                pyb.resetBasePositionAndOrientation(vectors[idx], (x, y, h),
                                                    Quaternion.from_rotation(np.array((x_ax, y_ax, n)).T))
                idx += 1
        pyb.resetBasePositionAndOrientation(box_id, t.getPeakInRegion((-0.5, 0.5), (-0.5, 0.5)), (0., 0., 0., 1.))
    time.sleep(300)
