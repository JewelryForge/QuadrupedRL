from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

__all__ = ['ActuatorNet']


class ActuatorNet(nn.Module):
    activation = nn.Softsign

    def __init__(self, input_dim=3, output_dim=1, hidden_dims=(32, 32, 32)):
        super().__init__()
        layers = []
        self.input_dim, self.output_dim, self.hidden_dims = input_dim, output_dim, hidden_dims
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(self.activation())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)

    def calc_torque(self, error, error_rate, velocity):
        device = next(self.parameters()).device
        X = np.array((error, velocity, error_rate), dtype=np.float32).transpose()
        return self(torch.tensor(X).to(device)).detach().squeeze().cpu().numpy().astype(float)


class RobotDataset(Dataset):
    def __init__(self, path, max_size=None):
        self.data = np.load(path)
        if max_size:
            self.error = self.data['angle_error'][:max_size, :].flatten().astype(np.float32)
            self.velocity = self.data['motor_velocity'][:max_size, :].flatten().astype(np.float32)
            self.error_rate = self.data['angle_error_rate'][:max_size, :].flatten().astype(np.float32)
            self.torque = self.data['motor_torque'][:max_size, :].flatten().astype(np.float32)
        else:
            self.error = self.data['angle_error'].flatten().astype(np.float32)
            self.velocity = self.data['motor_velocity'].flatten().astype(np.float32)
            self.error_rate = self.data['angle_error_rate'].flatten().astype(np.float32)
            self.torque = self.data['motor_torque'].flatten().astype(np.float32)
        self.X = np.stack((self.error, self.velocity, self.error_rate), axis=1)
        self.Y = np.expand_dims(self.torque, axis=1)
        self.size = min(len(self.error), len(self.velocity), len(self.error_rate), len(self.torque))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_actuator_net(actuator_net, dataset, lr=1e-3, num_epochs=1000, batch_size=1000, device='cuda'):
    wandb.init(project='actuator_net', name=str((actuator_net.hidden_dims, lr)), mode=None)
    device = torch.device(device)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_data, test_data = random_split(dataset, (train_len, test_len))
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    net = actuator_net.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    init_logger()
    log_dir = f'ident/{timestamp()}'
    os.makedirs(log_dir, exist_ok=True)
    torch.set_printoptions(linewidth=10000)
    for i in range(1, num_epochs + 1):
        train_loss, test_loss = 0., 0.
        for X, Y in train_loader:
            optim.zero_grad()
            X, Y = X.to(device), Y.to(device)
            loss = criterion(net(X), Y)
            train_loss += loss.item()
            loss.backward()
            optim.step()
        train_loss /= train_len
        log_info(f'Epoch {i:>4}/{num_epochs} train loss {np.mean(train_loss):.6f}')

        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            loss = criterion(net(X), Y)
            test_loss += loss.item()
        test_loss /= test_len
        log_info(f'Epoch {i:>4}/{num_epochs} test loss {np.mean(test_loss):.6f}')
        wandb.log({'Train/loss': train_loss, 'Test/loss': test_loss})
        if i % 10 == 0:
            torch.save({'model': net.state_dict(), 'hidden_dims': net.hidden_dims},
                       os.path.join(log_dir, f'{i}.pt'))


if __name__ == '__main__':
    import sys
    import os
    from os.path import dirname, abspath
    import wandb

    sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
    import burl
    from burl.utils import timestamp, log_info, init_logger

    np.set_printoptions(3, linewidth=10000, suppress=True)

    dataset = RobotDataset('/home/jewel/state_cmd_data_281557.npz')
    device = torch.device('cuda')
    actuator_net = ActuatorNet(hidden_dims=(16, 16, 16))
    train = False
    if train:
        train_actuator_net(actuator_net, dataset, lr=1e-4, num_epochs=1000, device=device)
    else:
        actuator_net = actuator_net.to(device)
        model_path = os.path.join(burl.rsc_path, 'actuator_net.pt')
        actuator_net.load_state_dict(torch.load(model_path)['model'])

    motor_idx = 0
    error = dataset.data['angle_error'][:, motor_idx]
    error_rate = dataset.data['angle_error_rate'][:, motor_idx]
    velocity = dataset.data['motor_velocity'][:, motor_idx]
    torque = dataset.data['motor_torque'][:, motor_idx]
    predicted = actuator_net.calc_torque(error, error_rate, velocity)

    criterion = nn.MSELoss(reduction='sum')
    test_loader = DataLoader(dataset, 10, shuffle=True)
    test_loss = 0.
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        loss = criterion(actuator_net(X), Y)
        test_loss += loss.item()
    print(test_loss, test_loss / len(dataset))
    import matplotlib.pyplot as plt

    plt.subplot(4, 1, 1)
    plt.plot(error)
    plt.ylabel('error')
    plt.subplot(4, 1, 2)
    plt.plot(error_rate)
    plt.ylabel('error_rate')
    plt.subplot(4, 1, 3)
    plt.plot(velocity)
    plt.ylabel('velocity')
    plt.subplot(4, 1, 4)
    plt.plot(torque, linewidth=1)
    plt.ylabel('torque')
    plt.plot(predicted, 'r', linewidth=0.3)
    plt.show()
