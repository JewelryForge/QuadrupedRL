import copy

import torch
from torch import nn

from burl.alg import Actor
from burl.alg.tcn import TCNEncoder


class Student(nn.Module):
    def __init__(self, teacher: Actor):
        super().__init__()
        self.locomotion_layers = copy.deepcopy(teacher.locomotion_layers).requires_grad_(False)
        self.action_layers = copy.deepcopy(teacher.action_layers).requires_grad_(False)
        self.history_encoder = TCNEncoder()

    def forward(self, proprio_history, real_world_obs):
        extero_feature = self.history_encoder(proprio_history)
        locomotion_feature = self.locomotion_layers(real_world_obs)
        return self.action_layers(torch.concat((extero_feature, locomotion_feature), dim=-1))

    def get_encoded(self, proprio_history, real_world_obs):
        extero_feature = self.history_encoder(proprio_history)
        locomotion_feature = self.locomotion_layers(real_world_obs)
        action = self.action_layers(torch.concat((extero_feature, locomotion_feature), dim=-1))
        return extero_feature, action
