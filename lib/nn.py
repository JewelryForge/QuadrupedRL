import torch
from torch import nn
from state import ProprioceptiveObservation, PrivilegedInformation, Observation, Action


class Teacher(nn.Module):
    NUM_FEATURES = (72, 64, 256, 128, 64)

    def __init__(self):
        super(Teacher, self).__init__()
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

    def forward(self, obs: torch.Tensor, pi: torch.Tensor):
        features = self.encoder(pi)
        generalized = torch.concat((obs, features), dim=1)
        return self.decoder(generalized)


if __name__ == '__main__':
    net = Teacher().to("cuda")
    from torchsummary import summary

    summary(net, [(Observation.dim,), (PrivilegedInformation.dim,)])
