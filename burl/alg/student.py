import copy

import torch
from torch import nn

from burl.alg import Actor
from burl.alg.tcn import TCNEncoder


class Student(nn.Module):
    def __init__(self, teacher: Actor):
        super().__init__()
        self.proprio_layers = copy.deepcopy(teacher.proprio_layers).requires_grad_(False)
        self.action_layers = copy.deepcopy(teacher.action_layers).requires_grad_(False)
        self.encoder = TCNEncoder()

    def forward(self, proprio_obs, history):
        estimated_extero_feature, proprio_feature = self.encoder(history), self.proprio_layers(proprio_obs)
        return self.action_layers(torch.concat((estimated_extero_feature, proprio_feature), dim=-1))

    def feedback(self, proprio_obs, history):
        estimated_extero_feature, proprio_feature = self.encoder(history), self.proprio_layers(proprio_obs)
        action = self.action_layers(torch.concat((estimated_extero_feature, proprio_feature), dim=-1))
        return estimated_extero_feature, action


