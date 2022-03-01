import math
import numpy as np


def ang_norm(x):  # [-pi, pi)
    return x - ((x + math.pi) // math.tau) * math.tau


def unit(x) -> np.ndarray:
    return np.asarray(x) / math.hypot(*x)


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
