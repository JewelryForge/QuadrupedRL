from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class TgNet(nn.Module):
    activation = nn.Tanh

    def __init__(self, input_dim=2, output_dim=3, hidden_dims=(64, 128)):
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


class WholeBodyTgNet(nn.Module):
    input_dim = 8
    action_dim = 12

    def __init__(self, init_model_dir=None):
        super().__init__()
        if init_model_dir:
            init_net = TgNet()
            init_net.load_state_dict(torch.load(init_model_dir)['model'])
            self.nets = nn.ModuleList(copy.deepcopy(init_net) for _ in range(4))
        else:
            self.nets = nn.ModuleList(TgNet() for _ in range(4))

    def forward(self, X):  # shape: (..., 8) -> (..., 12)
        inputs = [X[..., 2 * i:2 * i + 2] for i in range(4)]
        return torch.cat([net(x) for net, x in zip(self.nets, inputs)], dim=-1)


def implicit_tg(init_model_dir=None, device='cuda'):
    device = torch.device(device)
    tg_net = WholeBodyTgNet(init_model_dir).to(device)

    def _tg(phases):
        phases = torch.tensor(phases, dtype=torch.float).to(device)
        X = torch.stack((torch.sin(phases), torch.cos(phases)), dim=1).reshape(-1, 8)
        return tg_net(X).detach().cpu().numpy().reshape(-1)

    return _tg


if __name__ == '__main__':
    import sys
    import os

    sys.path.append('D:/Workspaces/teacher-student')
    from matplotlib import pyplot as plt
    import wandb
    from burl.sim.tg import vertical_tg
    from burl.utils import timestamp

    hidden_dims = eval(sys.argv[1])
    closed_tg = vertical_tg(h=0.12)
    wandb.init(project='tg-fit', name=str(hidden_dims), mode=None)
    device = torch.device('cuda')
    net = TgNet(hidden_dims=hidden_dims).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    num_epochs = 30000
    batch_size = 1000
    log_dir = f'tg_log/{timestamp()}'
    os.makedirs(log_dir, exist_ok=True)
    torch.set_printoptions(linewidth=10000)
    try:
        for i in range(1, num_epochs + 1):
            phases = np.random.uniform(-np.pi, np.pi, batch_size)
            # coords = np.array([np.tile(closed_tg(phi), 4) for phi in phases])
            coords = closed_tg(phases)
            phases = torch.tensor(phases, dtype=torch.float).to(device).unsqueeze(dim=1)
            X = torch.cat((torch.sin(phases), torch.cos(phases)), dim=1)
            Y = torch.tensor(coords, dtype=torch.float).to(device)
            optim.zero_grad()
            loss = criterion(net(X), Y)
            loss.backward()
            optim.step()
            wandb.log({'Train/loss': loss})
            print(f'Epoch {i:>4}/{num_epochs} loss {loss:.6f}')
            if i % 1000 == 0:
                torch.save({'model': net.state_dict()}, os.path.join(log_dir, f'{i}.pt'))
    except KeyboardInterrupt:
        pass
        # print(net(X))
        # print(X.shape, Y.shape)
        # raise RuntimeError

    phi = np.linspace(-np.pi, np.pi, 1000, dtype=np.float32)
    phi_t = torch.tensor(phi, requires_grad=False).to(device).unsqueeze(dim=1)
    X = torch.cat((torch.sin(phi_t), torch.cos(phi_t)), dim=1)
    y = net(X)
    plt.plot(phi, y.cpu().detach().numpy())
    plt.title(str(hidden_dims))
    plt.legend(['x1', 'y1', 'z1'])
    # plt.legend(['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4'])
    plt.show()
