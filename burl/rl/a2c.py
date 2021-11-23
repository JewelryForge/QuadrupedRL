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


if __name__ == '__main__':
    t = ActorTeacher()
    c = Critic()
