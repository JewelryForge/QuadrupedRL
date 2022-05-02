import copy
import os.path

import torch
from torch import nn

from burl.alg.ac import Actor
from burl.alg.tcn import TCNEncoder, TCNEncoderNoPadding


class Student(nn.Module):
    def __init__(self, teacher: Actor, encoder_type=TCNEncoderNoPadding):
        super().__init__()
        self.real_world_obs_dim = teacher.real_world_obs_dim
        self.encoder_input_dim = encoder_type.input_dim
        self.locomotion_layers = copy.deepcopy(teacher.locomotion_layers).requires_grad_(False)
        self.action_layers = copy.deepcopy(teacher.action_layers).requires_grad_(False)
        self.history_encoder = encoder_type()

    def forward(self, proprio_history, real_world_obs):
        extero_feature = self.history_encoder(proprio_history)
        locomotion_feature = self.locomotion_layers(real_world_obs)
        return self.action_layers(torch.concat((extero_feature, locomotion_feature), dim=-1))

    def get_encoded(self, proprio_history, real_world_obs):
        extero_feature = self.history_encoder(proprio_history)
        locomotion_feature = self.locomotion_layers(real_world_obs)
        action = self.action_layers(torch.concat((extero_feature, locomotion_feature), dim=-1))
        return extero_feature, action.tanh()

    def get_policy(self):
        return lambda *x: self(*x).tanh()

    def save_deployable_model(self, path=''):
        device = next(self.parameters()).device
        realworld_example = torch.rand(1, self.real_world_obs_dim, device=device)
        proprio_history_example = torch.rand(1, *self.encoder_input_dim, device=device)
        script_module = torch.jit.trace(self, (proprio_history_example, realworld_example))
        print(script_module)
        if not path:
            from burl.exp import student_log_dir
            path = os.path.join(student_log_dir, 'script_model.pt')
        torch.jit.save(script_module, path)
