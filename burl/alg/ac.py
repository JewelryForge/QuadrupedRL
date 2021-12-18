import torch
from torch import nn

from burl.rl.state import PrivilegedInformation, Observation, Action, ExtendedObservation
from burl.utils import g_cfg


class ActorTeacher(nn.Module):
    NUM_FEATURES = (72, 64, 256, 128, 64)
    INPUT_DIM = ExtendedObservation.dim
    OUTPUT_DIM = Action.dim
    activation = nn.Tanh

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PrivilegedInformation.dim, self.NUM_FEATURES[0]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[0], self.NUM_FEATURES[1]),
            self.activation()
        )

        self.decoder = nn.Sequential(
            nn.Linear(Observation.dim + self.NUM_FEATURES[1], self.NUM_FEATURES[2]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[2], self.NUM_FEATURES[3]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[3], self.NUM_FEATURES[4]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[4], self.OUTPUT_DIM)
        )

    def forward(self, x):
        pinfo, obs = x[:, :PrivilegedInformation.dim], x[:, PrivilegedInformation.dim:]
        features = self.encoder(pinfo)
        generalized = torch.concat((obs, features), dim=1)
        return self.decoder(generalized)


class Critic(nn.Module):
    NUM_FEATURES = (256, 128, 64)
    INPUT_DIM = ExtendedObservation.dim
    OUTPUT_DIM = 1
    activation = nn.ELU

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ExtendedObservation.dim, self.NUM_FEATURES[0]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[0], self.NUM_FEATURES[1]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[1], self.NUM_FEATURES[2]),
            self.activation(),
            nn.Linear(self.NUM_FEATURES[2], 1)
        )

    def forward(self, x):
        return self.layers(x)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, actor: ActorTeacher, critic: Critic):
        super(ActorCritic, self).__init__()

        # Policy and value function
        self.actor, self.critic = actor, critic

        # Action noise
        self.std = nn.Parameter(g_cfg.init_noise_std * torch.ones(actor.OUTPUT_DIM))
        self.distribution = None
        torch.distributions.Normal.set_default_validate_args = False

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
        return actions_mean.detach()

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


if __name__ == '__main__':
    t = ActorTeacher()
    c = Critic()
