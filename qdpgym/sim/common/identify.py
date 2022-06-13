import numpy as np
import torch
from torch import nn

__all__ = ['ActuatorNet']


class ActuatorNet(nn.Module):
    activation = nn.Softsign

    def __init__(
        self,
        input_dim=6,
        output_dim=1,
        hidden_dims=(32, 32, 32)
    ):
        super().__init__()
        layers = []
        self.input_dim, self.output_dim, self.hidden_dims = input_dim, output_dim, hidden_dims
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.device = torch.device('cpu')

    def forward(self, state):
        return self.layers(state)

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return super().to(device, *args, **kwargs)

    def calc_torque(self, err1, err2, err3, vel1, vel2, vel3):
        with torch.inference_mode():
            X = np.array(
                (err1, err2, err3, vel1, vel2, vel3),
                dtype=np.float32
            ).transpose()
            try:
                Y = self(torch.as_tensor(X, device=self.device))
            except RuntimeError:
                self.device = next(self.parameters()).device
                Y = self(torch.as_tensor(X, device=self.device))
            return Y.double().squeeze().cpu().numpy()
