import numpy as np
from scipy.spatial.transform import Rotation as scipyRotation


class NDArrayBased(np.ndarray):
    def __new__(cls, matrix, skip_check=False):
        matrix = np.asarray(matrix)
        if skip_check:
            return matrix.view(cls)
        if cls.is_valid(matrix):
            return cls.preprocess(matrix).view(cls)
        raise RuntimeError(f'Invalid {cls}')

    @classmethod
    def is_valid(cls, matrix):
        return True

    @classmethod
    def preprocess(cls, matrix):
        return matrix


class Rotation(NDArrayBased):
    THRESHOLD = 1e-5

    @classmethod
    def is_valid(cls, matrix):
        matrix = np.asarray(matrix)
        if matrix.shape != (3, 3):
            return False
        matrix_t = np.transpose(matrix)
        error = np.eye(3) - np.dot(matrix_t, matrix)
        return np.linalg.norm(error) < 1e-6

    @classmethod
    def from_rpy(cls, rpy):  # zyx
        sr, sp, sy = np.sin(rpy)
        cr, cp, cy = np.cos(rpy)
        _matrix = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                   (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                   (-sp, cp * sr, cp * cr))
        return cls(_matrix, skip_check=True)

    @classmethod
    def from_quaternion(cls, q):
        if not Quaternion.is_valid(q):
            raise RuntimeError('Invalid Quaternion')
        x, y, z, w = q
        xx, xy, xz, xw = x * x, x * y, x * z, x * w
        yy, yz, yw, zz, zw, ww = y * y, y * z, y * w, z * z, z * w, w * w
        _matrix = ((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                   (2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                   (2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy))
        return cls(_matrix, skip_check=True)

    def inverse(self):
        return self.transpose()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (f'[{self[0, 0]: .3f} {self[0, 1]: .3f} {self[0, 2]: .3f} ]\n'
                f'[{self[1, 0]: .3f} {self[1, 1]: .3f} {self[1, 2]: .3f} ]\n'
                f'[{self[2, 0]: .3f} {self[2, 1]: .3f} {self[2, 2]: .3f} ]\n')

    def __setitem__(self, key, value):
        raise RuntimeError('Cannot Set Item Directly')

    @property
    def X(self):
        return self[:, 0].view(np.ndarray)

    @property
    def Y(self):
        return self[:, 1].view(np.ndarray)

    @property
    def Z(self):
        return self[:, 2].view(np.ndarray)


ARR_ZERO3 = np.zeros(3)
ARR_EYE3 = np.eye(3)


class Odometry(object):
    def __init__(self, rotation=ARR_EYE3, translation=ARR_ZERO3):
        self.rotation, self.translation = np.asarray(rotation), np.asarray(translation)

    def multiply(self, other):
        if isinstance(other, Odometry):
            return Odometry(self.rotation @ other.rotation,
                            self.translation + self.rotation @ other.translation)
        if isinstance(other, np.ndarray):
            return self.rotation @ other + self.translation
        raise RuntimeError(f"Unsupported datatype '{type(other)}' for multiplying")

    def __matmul__(self, other):
        return self.multiply(other)

    def __imatmul__(self, other):
        self.rotation @= other.rotation
        self.translation += self.rotation @ other.translation

    def __repr__(self):
        return str(np.concatenate((self.rotation, np.expand_dims(self.translation, axis=1)), axis=1))


class Quaternion(NDArrayBased):
    THRESHOLD = 1e-5

    @classmethod
    def is_valid(cls, matrix):
        matrix = np.asarray(matrix)
        return matrix.squeeze().shape == (4,) and abs(np.linalg.norm(matrix) - 1) < cls.THRESHOLD

    @classmethod
    def preprocess(cls, matrix):
        return matrix.squeeze()

    @classmethod
    def identity(cls):
        return cls((0, 0, 0, 1))

    @classmethod
    def from_rpy(cls, rpy):
        rpy = np.asarray(rpy)
        sr, sp, sy = np.sin(rpy / 2)
        cr, cp, cy = np.cos(rpy / 2)
        return cls((sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy), skip_check=True)

    @classmethod
    def from_rotation(cls, r):
        return scipyRotation.from_matrix(r).as_quat()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'[{self[0]: .3f} {self[1]: .3f} {self[2]: .3f} {self[3]: .3f} ]'

    @property
    def x(self):
        return self[0].view(float)

    @property
    def y(self):
        return self[1].view(float)

    @property
    def z(self):
        return self[2].view(float)

    @property
    def w(self):
        return self[3].view(float)


class Rpy(NDArrayBased):
    @classmethod
    def preprocess(cls, matrix):
        return matrix.squeeze()

    @classmethod
    def from_quaternion(cls, q):
        if not Quaternion.is_valid(q):
            raise RuntimeError('Invalid Quaternion')
        x, y, z, w = q
        return cls((np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                    np.arcsin(2 * (w * y - x * z)),
                    np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), skip_check=True)

    @classmethod
    def from_rotation(cls, r):
        return scipyRotation.from_matrix(r).as_euler('ZYX', degrees=False)[::-1]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'[{self[0]: .3f} {self[1]: .3f} {self[2]: .3f} ]'

    @property
    def r(self):
        return self[0].view(float)

    @property
    def p(self):
        return self[1].view(float)

    @property
    def y(self):
        return self[2].view(float)


def get_rpy_rate_from_angular_velocity(rpy, angular):
    r, p, y = rpy
    sp, cp = np.sin(p), np.cos(p)
    cr, tr = np.cos(r), np.tan(r)
    trans = np.array(((1, sp * tr, cp * tr),
                      (0, cp, -sp),
                      (0, sp / cr, cp / cr)))
    return np.dot(trans, angular)


if __name__ == '__main__':
    from burl.utils import WithTimer

    r = np.array(Rotation.from_rpy((0.16, 0.28, 0.4)))
    for _ in range(10):
        with WithTimer() as timer:
            Rpy.from_rotation(r)
        print(timer.time_spent)
