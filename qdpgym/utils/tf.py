import math
from typing import Tuple, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation as scipyRotation

__all__ = ['Rotation', 'Quaternion', 'Rpy', 'Odometry',
           'vnorm', 'vunit', 'vcross', 'estimate_normal',
           'get_rpy_rate_from_ang_vel']


def vnorm(vec):
    return math.sqrt(sum(x ** 2 for x in vec))


def vunit(x) -> np.ndarray:
    return np.asarray(x) / vnorm(x)


def vcross(vec3_1, vec3_2) -> np.ndarray:
    """
    A much faster alternative for np.cross.
    """
    (x1, y1, z1), (x2, y2, z2) = vec3_1, vec3_2
    return np.array((y1 * z2 - y2 * z1, z1 * x2 - z2 * x1, x1 * y2 - x2 * y1))


class Rotation(object):
    THRESHOLD = 1e-5

    @staticmethod
    def from_rpy(rpy):  # zyx
        sr, sp, sy = np.sin(rpy)
        cr, cp, cy = np.cos(rpy)
        matrix = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                  (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                  (-sp, cp * sr, cp * cr))
        return np.array(matrix)

    @staticmethod
    def from_quaternion(q):
        return scipyRotation.from_quat(q).as_matrix()

    @staticmethod
    def from_zaxis(z):
        if z[2] == 1.0:
            ref = np.array((1., 0., 0.))
        else:
            ref = np.array((0., 0., 1.))
        x = vcross(z, ref)
        x /= vnorm(x)
        y = vcross(z, x)
        return np.array((x, y, z)).T


class Quaternion(object):
    """
    q = [x y z w] = w + xi + yj + zk
    """

    @staticmethod
    def identity():
        return np.array((0., 0., 0., 1.))

    @staticmethod
    def from_rpy(rpy):
        rpy_2 = np.asarray(rpy) / 2
        (sr, sp, sy), (cr, cp, cy) = np.sin(rpy_2), np.cos(rpy_2)
        return np.array(((sr * cp * cy - cr * sp * sy,
                          cr * sp * cy + sr * cp * sy,
                          cr * cp * sy - sr * sp * cy,
                          cr * cp * cy + sr * sp * sy)))

    @staticmethod
    def from_rotation(r):
        # scipy api is faster than python implementation
        return scipyRotation.from_matrix(r).as_quat()

    @staticmethod
    def inverse(q):
        x, y, z, w = q
        return np.array((-x, -y, -z, w))

    @staticmethod
    def from_wxyz(q):
        w, x, y, z = q
        return np.array((x, y, z, w))

    @staticmethod
    def to_wxyz(q):
        x, y, z, w = q
        return np.array((w, x, y, z))


class Rpy(object):
    """ZYX intrinsic Euler Angles (roll, pitch, yaw)"""

    @staticmethod
    def from_quaternion(q):
        x, y, z, w = q
        return np.array((np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                         np.arcsin(2 * (w * y - x * z)),
                         np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))

    @classmethod
    def from_rotation(cls, r):
        return scipyRotation.from_matrix(r).as_euler('ZYX', degrees=False)[::-1]


def get_rpy_rate_from_ang_vel(rpy, angular):
    r, p, y = rpy
    sp, cp = math.sin(p), math.cos(p)
    cr, tr = math.cos(r), math.tan(r)
    trans = np.array(((1, sp * tr, cp * tr),
                      (0, cp, -sp),
                      (0, sp / cr, cp / cr)))
    return np.dot(trans, angular)


FLOAT3 = Tuple[float, float, float]


def estimate_normal(points: Union[Sequence[FLOAT3], np.ndarray]):
    """
    Estimate the normal of terrain from a set of points.
    The terrain CANNOT be vertical.
    :param points: a set of terrain points
    :return: np.ndarray, the normal vector
    """
    X, Y, Z = np.array(points).T
    A = np.zeros((3, 3))
    A[0, :] = np.sum(X ** 2), X @ Y, np.sum(X)
    A[1, :] = A[0, 1], np.sum(Y ** 2), np.sum(Y)
    A[2, :] = A[0, 2], A[1, 2], len(X)
    b = np.array((X @ Z, Y @ Z, np.sum(Z)))
    a, b, _ = np.linalg.solve(A, b)
    return vunit((-a, -b, 1))


ARR_ZERO3 = (0., 0., 0.)
ARR_EYE3 = (ARR_ZERO3,) * 3


class Odometry(object):
    """
    Homogeneous coordinate transformation defined by a rotation and a translation:
        ((R_3x3, t_1x3)
         (0_3x1, 1_1x1))
    """

    def __init__(self, rotation=ARR_EYE3, translation=ARR_ZERO3):
        self.rotation, self.translation = np.asarray(rotation), np.asarray(translation)

    def multiply(self, other):
        if isinstance(other, Odometry):
            return Odometry(self.rotation @ other.rotation,
                            self.translation + self.rotation @ other.translation)
        if isinstance(other, np.ndarray):
            return self.rotation @ other + self.translation
        raise RuntimeError(f'Unsupported datatype `{type(other)}` for multiplying')

    def __matmul__(self, other):
        return self.multiply(other)

    def __imatmul__(self, other):
        self.rotation @= other.rotation
        self.translation += self.rotation @ other.translation

    def __repr__(self):
        return str(np.concatenate((self.rotation, np.expand_dims(self.translation, axis=1)), axis=1))
