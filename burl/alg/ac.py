from collections.abc import Iterable
import math
import torch
from torch import nn


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class Actor(nn.Module):
    activation = nn.Tanh

    def __init__(self, extero_obs_dim, proprio_obs_dim, action_dim,
                 extero_layer_dims=(72, 64),
                 proprio_layer_dims=(),
                 action_layer_dims=(256, 128, 64),
                 init_noise_std=0.1):
        super().__init__()
        self.input_dim = extero_obs_dim + proprio_obs_dim
        self.extero_obs_dim, self.proprio_obs_dim, self.output_dim = extero_obs_dim, proprio_obs_dim, action_dim
        extero_layers, proprio_layers, action_layers = [], [], []
        self.extero_obs_dim = extero_feature_dim = extero_obs_dim
        if extero_layer_dims:
            for dim in extero_layer_dims:
                extero_layers.append(nn.Linear(extero_feature_dim, dim))
                extero_layers.append(self.activation())
                extero_feature_dim = dim

        self.proprio_obs_dim = proprio_feature_dim = proprio_obs_dim
        if proprio_layer_dims:
            for dim in extero_layer_dims:
                proprio_layers.append(nn.Linear(proprio_feature_dim, dim))
                proprio_layers.append(self.activation())
                proprio_feature_dim = dim

        action_feature_dim = extero_feature_dim + proprio_feature_dim
        for dim in action_layer_dims:
            action_layers.append(nn.Linear(action_feature_dim, dim))
            action_layers.append(self.activation())
            action_feature_dim = dim
        action_layers.append(nn.Linear(action_feature_dim, action_dim))

        self.extero_layers = nn.Sequential(*extero_layers)
        self.proprio_layers = nn.Sequential(*proprio_layers)
        self.action_layers = nn.Sequential(*action_layers)

        layer_norm(self.action_layers[-1], std=0.1)  # output layer for action

        if isinstance(init_noise_std, Iterable):
            self.log_std = nn.Parameter(torch.Tensor(init_noise_std).log(), requires_grad=True)
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std), dtype=torch.float32),
                                        requires_grad=True)

        # if isinstance(init_noise_std, Iterable):
        #     self.std = nn.Parameter(torch.Tensor(init_noise_std), requires_grad=True)
        # else:
        #     self.std = nn.Parameter(torch.full((action_dim,), init_noise_std, dtype=torch.float32), requires_grad=True)

        self.distribution = None

    def forward(self, x):
        extero_obs, proprio_obs = x[:, :self.extero_obs_dim], x[:, self.extero_obs_dim:]
        extero_features, proprio_features = self.extero_layers(extero_obs), self.proprio_layers(proprio_obs)
        return self.action_layers(torch.concat((extero_features, proprio_features), dim=-1)).tanh()

    @property
    def std(self):
        return self.log_std.exp()

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_action(self, actor_obs):
        mean = self(actor_obs)
        self.distribution = torch.distributions.Normal(mean, self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)


class Critic(nn.Module):
    activation = nn.ELU

    def __init__(self, input_dim, output_dim=1, hidden_dims=(256, 256, 256)):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        layer_norm(self.layers[-1], std=0.5)

    def forward(self, state):
        return self.layers(state)


if __name__ == '__main__':
    from burl.rl.state import ExteroObservation, ProprioObservation, Action, ExtendedObservation

    actor = Actor(ExteroObservation.dim, ProprioObservation.dim, Action.dim)
    critic = Critic(ExtendedObservation.dim)
    print(actor, critic, sep='\n')
    print(actor(torch.unsqueeze(torch.Tensor(ExtendedObservation().to_array()), dim=0)))
    print(critic(torch.unsqueeze(torch.Tensor(ExtendedObservation().to_array()), dim=0)))
