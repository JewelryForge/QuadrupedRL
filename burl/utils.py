import pybullet as p
import numpy as np


def normalize(x):
    return x - ((x + np.pi) // (2 * np.pi)) * (2 * np.pi)


def unit(x: np.ndarray):
    return x / np.linalg.norm(x)


def tuple_compact_string(_tuple):
    return '(' + ' '.join(f'{f:.1f}' for f in _tuple) + ')'


def make_class(cls, **properties):
    def _make_class(*args, **kwargs):
        properties.update(**kwargs)
        return cls(*args, **properties)

    return _make_class


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


if __name__ == '__main__':
    c = make_class(JointInfo)
    print(c.__closure__[0].cell_contents, c.__closure__[1].cell_contents)
