import torch
import numpy as np
from torch import nn

M_PI = np.pi
M_2_PI = 2 * M_PI

def normalize(x):
    return x - ((x + M_PI) // M_2_PI) * M_2_PI


def unit(x: np.ndarray):
    return x / np.linalg.norm(x)


def tuple_compact_string(_tuple):
    return '(' + ' '.join(f'{f:.1f}' for f in _tuple) + ')'


def make_cls(cls, **properties):
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


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


activations = {
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'relu': nn.ReLU(),
    'lrelu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softsign': nn.Softsign()
}


def get_activation(activation_name):
    try:
        return activations[activation_name]
    except KeyError:
        raise RuntimeError("invalid activation function!")
