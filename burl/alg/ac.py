import torch
from torch import nn


class Actor(nn.Module):
    activation = nn.Tanh

    def __init__(self, extero_obs_dim, proprio_obs_dim, action_dim,
                 extero_layer_dims=(72, 64),
                 proprio_layer_dims=(),
                 action_layer_dims=(256, 128, 64)):
        super().__init__()
        self.extero_obs_dim, self.proprio_obs_dim, self.action_dim = extero_obs_dim, proprio_obs_dim, action_dim
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

    def forward(self, x):
        extero_obs, proprio_obs = x[:, :self.extero_obs_dim], x[:, self.extero_obs_dim:]
        extero_features, proprio_features = self.extero_layers(extero_obs), self.proprio_layers(proprio_obs)
        return self.action_layers(torch.concat((extero_features, proprio_features), dim=-1))


class Critic(nn.Module):
    activation = nn.ELU

    def __init__(self, input_dim, output_dim=1, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, actor: Actor, critic: Critic, init_noise_std):
        super(ActorCritic, self).__init__()

        # Policy and value function
        self.actor, self.critic = actor, critic

        # Action noise
        # self.log_std = nn.Parameter(torch.full(actor.action_dim, torch.log(init_noise_std)), requires_grad=True)
        # self.std = torch.exp(self.log_std)
        self.std = nn.Parameter(torch.full(16, init_noise_std), requires_grad=True)
        self.distribution = None
        torch.distributions.Normal.set_default_validate_args = False

        print('Actor:', self.actor, 'Critic:', self.critic, sep='\n')

    @staticmethod
    def init_weights(sequential, scales):  # not used at the moment
        for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, **kwargs):
        mean = self.actor(observations)
        # self.std = torch.exp(self.log_std)
        # self.distribution = torch.distributions.Normal(mean, self.std)
        self.distribution = torch.distributions.Normal(mean, torch.clip(self.std, 0.01))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean.detach()

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


if __name__ == '__main__':
    from burl.rl.state import ExteroObservation, ProprioObservation, Action, ExtendedObservation

    actor = Actor(ExteroObservation.dim, ProprioObservation.dim, Action.dim)
    critic = Critic(ExtendedObservation.dim)
    print(actor, critic, sep='\n')
    print(actor(torch.unsqueeze(torch.Tensor(ExtendedObservation().to_array()), dim=0)))
    print(critic(torch.unsqueeze(torch.Tensor(ExtendedObservation().to_array()), dim=0)))
