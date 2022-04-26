import math
from collections.abc import Sequence
from functools import singledispatchmethod

import numpy as np

PI, TAU = math.pi, math.tau


class Angle(object):
    @singledispatchmethod
    @staticmethod
    def norm(x):
        if -PI <= x < PI:
            return x
        return x - int((x + PI) / TAU) * TAU

    @norm.register(np.ndarray)
    @staticmethod
    def _(x: np.ndarray):
        return x - ((x + PI) / TAU).astype(int) * TAU

    @staticmethod
    def mean(lst: Sequence[float]):
        _sum = last = lst[0]
        for a in lst[1:]:
            last = Angle.near(last, a)
            _sum += last
        return Angle.norm(_sum / len(lst))

    @staticmethod
    def near(x, y):
        if x > y + PI:
            y += TAU
        elif x < y - PI:
            y -= TAU
        return y


ang_norm = Angle.norm


def norm(vec):
    return math.hypot(*vec)


def unit(x) -> np.ndarray:
    return np.asarray(x) / norm(x)


def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def vec_cross(vec3_1, vec3_2) -> np.ndarray:
    """
    A much faster alternative for np.cross.
    """
    (x1, y1, z1), (x2, y2, z2) = vec3_1, vec3_2
    return np.array((y1 * z2 - y2 * z1, z1 * x2 - z2 * x1, x1 * y2 - x2 * y1))


def clip(value: float, lower: float, upper: float) -> float:
    return upper if value > upper else lower if value < lower else value


def safe_asin(value):
    return math.asin(clip(value, -1., 1.))


def safe_acos(value):
    return math.acos(clip(value, -1., 1.))


def included_angle(vec1, vec2) -> float:
    return safe_acos(np.dot(unit(vec1), unit(vec2)))
