import enum

import torch

from .utils import tf

try:
    from torch import inference_mode as _im
except ImportError:
    torch.inference_mode = torch.no_grad


class Sim(enum.IntEnum):
    BULLET = 0
    MUJOCO = 1


sim_engine: Sim = Sim.BULLET
