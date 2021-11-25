import torch
from torch import nn

from burl.rl.state import PrivilegedInformation, Observation, Action, ExtendedObservation


class ActorTeacher(nn.Module):
    NUM_FEATURES = (72, 64, 256, 128, 64)

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PrivilegedInformation.dim, self.NUM_FEATURES[0]),
            nn.Tanh(),
            nn.Linear(self.NUM_FEATURES[0], self.NUM_FEATURES[1]),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(Observation.dim + self.NUM_FEATURES[1], self.NUM_FEATURES[2]),
            nn.Tanh(),
            nn.Linear(self.NUM_FEATURES[2], self.NUM_FEATURES[3]),
            nn.Tanh(),
            nn.Linear(self.NUM_FEATURES[3], self.NUM_FEATURES[4]),
            nn.Tanh(),
            nn.Linear(self.NUM_FEATURES[4], Action.dim)
        )

    def forward(self, x):
        pinfo, obs = x[:, :PrivilegedInformation.dim], x[:, PrivilegedInformation.dim:]
        features = self.encoder(pinfo)
        generalized = torch.concat((obs, features), dim=1)
        return self.decoder(generalized)


class Critic(nn.Module):
    NUM_FEATURES = (256, 128, 64)

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ExtendedObservation.dim, self.NUM_FEATURES[0]),
            nn.ELU(),
            nn.Linear(self.NUM_FEATURES[0], self.NUM_FEATURES[1]),
            nn.ELU(),
            nn.Linear(self.NUM_FEATURES[1], self.NUM_FEATURES[2]),
            nn.ELU(),
            nn.Linear(self.NUM_FEATURES[2], 1)
        )

    def forward(self, x):
        return self.layers(x)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, actor, critic, num_actions=16, init_noise_std=1.0, **kwargs):
        if kwargs:
            print("ActorCritic got unexpected arguments, which will be ignored:",
                  [key for key in kwargs.keys()])
        super(ActorCritic, self).__init__()

        # Policy and value function
        self.actor, self.critic = actor, critic

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        torch.distributions.Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    def init_weights(sequential, scales):  # not used at the moment
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = torch.distributions.Normal(mean, torch.clip(self.std, 0.01))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


if __name__ == '__main__':
    t = ActorTeacher()
    c = Critic()
