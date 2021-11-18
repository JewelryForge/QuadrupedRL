import numpy as np
from burl.utils.bc import NDArrayBased


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

    def __mul__(self, other):
        return np.dot(self, other).view(np.ndarray)

    def __rmul__(self, other):
        return np.dot(other, self).view(np.ndarray)

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


# class Odometry3d(object):
#     def __init__(self, ):
#         pass

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
        raise NotImplementedError

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
        raise NotImplementedError

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


if __name__ == '__main__':
    import pybullet

    r = np.random.rand(4)
    q = r / np.linalg.norm(r)
    print(q)
    ea = pybullet.getEulerFromQuaternion(q)
    print(ea)
    rq = Rpy.from_quaternion(q)
    print(rq)
# r = Rotation.from_rpy((np.pi / 6, 0, 0))  # * (0, 0, -1)
# print(np.squeeze(np.array([[1, 2, 3]]).transpose()).shape)
# print(r)
# print(r.Y)

# print(Rotation.is_valid([[1.00000000e+00, -3.25420248e-06, 3.38725419e-06],
#                            [-3.01673707e-06, 1.07793154e-01, 9.94173343e-01],
#                            [-3.60036417e-06, -9.94173343e-01, 1.07793154e-01]]))
#
# print(Rotation.from_rpy([0, 0, np.pi / 2]).matrix)
#
# r = Rotation.from_quaternion((-0.35, 1.23e-06, 4.18e-08, 0.39))
# print(r)
