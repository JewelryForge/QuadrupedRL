import os

import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import wandb

from burl.sim.tg import vertical_tg
from burl.utils import timestamp


class TGNet(nn.Module):
    activation = nn.Tanh

    def __init__(self, input_dim=2, output_dim=3, hidden_dims=(64, 32, 16)):
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


closed_tg = vertical_tg(h=0.12)

# wandb.init(project='tg-fit', mode='disabled')
wandb.init(project='tg-fit', mode=None)
device = torch.device('cuda')
net = TGNet().to(device)
optim = torch.optim.AdamW(net.parameters(), lr=1e-6)
criterion = nn.MSELoss()
num_epochs = 50000
batch_size = 1000
log_dir = f'tg_log/{timestamp()}'
os.mkdir(log_dir)
torch.set_printoptions(linewidth=10000)
for i in range(1, num_epochs + 1):
    phases = np.random.uniform(-np.pi, np.pi, batch_size)
    coords = np.array([closed_tg(phi) for phi in phases])
    phases = torch.tensor(phases, dtype=torch.float).to(device).unsqueeze(dim=1)
    X = torch.cat((torch.sin(phases), torch.cos(phases)), dim=1)
    Y = torch.tensor(coords, dtype=torch.float).to(device) * 0.1
    # print(Y.transpose(0, 1))
    # print(net(X).transpose(0, 1))
    # raise RuntimeError
    loss = criterion(net(X), Y)
    loss.backward()
    optim.step()
    wandb.log({'Train/loss': loss})
    print(f'Epoch {i:>4}/{num_epochs} loss {loss:.6f}')
    if i % 1000 == 0:
        torch.save({'model': net.state_dict()}, os.path.join(log_dir, f'{i}.pt'))
    # print(net(X))
    # print(X.shape, Y.shape)
    # raise RuntimeError

phi = np.linspace(-np.pi, np.pi, 1000, dtype=np.float32)
phi_t = torch.tensor(phi, requires_grad=False).to(device).unsqueeze(dim=1)
X = torch.cat((torch.sin(phi_t), torch.cos(phi_t)), dim=1)
y = net(X)
plt.plot(phi, y.cpu().detach().numpy())
plt.legend(['x', 'y', 'z'])
plt.show()
