import time
from datetime import datetime

import numpy as np

M_PI = np.pi
M_2_PI = 2 * M_PI


def normalize(x):  # [-pi, pi)
    return x - ((x + M_PI) // M_2_PI) * M_2_PI


def unit(x) -> np.ndarray:
    return np.asarray(x) / np.linalg.norm(x)


def vec_cross(X1, X2):
    """
    A much faster alternative for np.cross.
    """
    x1, y1, z1 = X1
    x2, y2, z2 = X2
    return np.array((y1 * z2 - y2 * z1, z1 * x2 - z2 * x1, x1 * y2 - x2 * y1))


def tuple_compact_string(_tuple):
    return '(' + ' '.join(f'{f:.1f}' for f in _tuple) + ')'


class make_cls(object):
    def __init__(self, cls, **properties):
        self.cls, self.properties = cls, properties

    def __call__(self, *args, **kwargs):
        properties = self.properties.copy()
        properties.update(**kwargs)
        return self.cls(*args, **properties)

    def __getattr__(self, item):
        return getattr(self.cls, item)


class WithTimer:
    def __init__(self):
        pass

    def start(self):
        self._start_time = time.time()

    def __enter__(self):
        self._start_time = time.time()
        return self

    def end(self):
        self._end_time = time.time()
        return self.time_spent

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()

    @property
    def time_spent(self):
        return self._end_time - self._start_time


class JointInfo(object):
    def __init__(self, joint_info: tuple):
        self._info = joint_info

    idx = property(lambda self: self._info[0])
    name = property(lambda self: self._info[1].decode("UTF-8"))
    type = property(lambda self: self._info[2])
    q_idx = property(lambda self: self._info[3])
    u_idx = property(lambda self: self._info[4])
    damping = property(lambda self: self._info[6])
    friction = property(lambda self: self._info[7])
    limits = property(lambda self: (self._info[8], self._info[9]))
    max_force = property(lambda self: self._info[10])
    max_vel = property(lambda self: self._info[11])
    link_name = property(lambda self: self._info[12].decode("UTF-8"))
    axis = property(lambda self: self._info[13])
    parent_frame_pos = property(lambda self: self._info[14])
    parent_frame_orn = property(lambda self: self._info[15])
    parent_idx = property(lambda self: self._info[16])

    def __str__(self):
        import pybullet as p
        analysis = []
        joint_types = {p.JOINT_REVOLUTE: 'revolute', p.JOINT_PRISMATIC: 'prismatic', p.JOINT_SPHERICAL: 'spherical',
                       p.JOINT_PLANAR: 'planar', p.JOINT_FIXED: 'fixed'}
        analysis.append(f'{self.idx}_{self.name}: {joint_types[self.type]}')
        is_fixed = self.type == p.JOINT_FIXED
        if not is_fixed:
            # analysis.append(f'q{joint_info[3]} u{joint_info[4]}')
            analysis.append(f'damp {self.damping} fric {self.friction}')
            analysis.append(f'[{self.limits[0]:.2f}, {self.limits[1]:.2f}]')
        analysis.append(self.link_name)
        if not is_fixed:
            axis_dict = {(1., 0., 0.): 'X', (0., 1., 0.): 'Y', (0., 0., 1.): 'Z',
                         (-1., 0., 0.): '-X', (0., -1., 0.): '-Y', (0., 0., -1.): '-Z', }
            if self.axis in axis_dict:
                analysis.append(f'axis {axis_dict[self.axis]}')
            else:
                analysis.append(f'axis{tuple_compact_string(self.axis)}')
        analysis.append(f'pos{tuple_compact_string(self.parent_frame_pos)}')
        analysis.append(f'orn{tuple_compact_string(self.parent_frame_orn)}')
        analysis.append(f'parent {self.parent_idx}')

        return ', '.join(analysis)


def timestamp():
    return datetime.now().strftime('%b%d_%H-%M-%S')


def str2time(time_str):
    try:
        return datetime.strptime(time_str, '%b%d_%H-%M-%S')
    except ValueError:
        return datetime(2000, 1, 1)


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


if __name__ == '__main__':
    t = timestamp()
    print(t)
    t1 = str2time(t)
    t = timestamp()
    t2 = str2time(t)
    print(t1 > t2)
