import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

__all__ = ['ActuatorNet', 'ActuatorNetWithHistory']


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
        self.device = torch.device('cuda')

    def forward(self, state):
        return self.layers(state)

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return super().to(device, *args, **kwargs)

    def calc_torque(self, error, error_rate, velocity):
        with torch.inference_mode():
            X = np.array((error, velocity, error_rate), dtype=np.float32).transpose()
            try:
                Y = self(torch.as_tensor(X, device=self.device))
            except RuntimeError:
                self.device = next(self.parameters()).device
                Y = self(torch.as_tensor(X, device=self.device))
            return Y.double().squeeze().cpu().numpy()


class ActuatorNetWithHistory(ActuatorNet):
    activation = nn.Softsign

    def __init__(self, input_dim=6, output_dim=1, hidden_dims=(32, 32, 32)):
        super().__init__(input_dim, output_dim, hidden_dims)

    def calc_torque(self, error, error_his1, error_his2,
                    velocity, velocity_his1, velocity_his2):
        with torch.inference_mode():
            X = np.array((error, error_his1, error_his2,
                          velocity, velocity_his1, velocity_his2), dtype=np.float32).transpose()
            try:
                Y = self(torch.as_tensor(X, device=self.device))
            except RuntimeError:
                self.device = next(self.parameters()).device
                Y = self(torch.as_tensor(X, device=self.device))
            return Y.double().squeeze().cpu().numpy()


class RobotDataset(Dataset):
    def __init__(self, path, slices=None):
        self.data = np.load(path)
        if slices:
            self.error = self.data['angle_error'][slices, :].astype(np.float32)
            self.velocity = self.data['motor_velocity'][slices, :].astype(np.float32)
            self.error_rate = self.data['angle_error_rate'][slices, :].astype(np.float32)
            self.torque = self.data['motor_torque'][slices, :].astype(np.float32)
        else:
            self.error = self.data['angle_error'].astype(np.float32)
            self.velocity = self.data['motor_velocity'].astype(np.float32)
            self.error_rate = self.data['angle_error_rate'].astype(np.float32)
            self.torque = self.data['motor_torque'].astype(np.float32)
        self.X = np.stack((self.error.flatten(), self.velocity.flatten(), self.error_rate.flatten()), axis=1)
        self.Y = np.expand_dims(self.torque.flatten(), axis=1)
        self.size = len(self.error)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class RobotDatasetWithHistory(Dataset):
    def __init__(self, path, history_interval=5, slices=None):
        print(path)
        self.data = np.load(path)
        if slices:
            self.error = self.data['angle_error'][slices, :].astype(np.float32)
            self.velocity = self.data['motor_velocity'][slices, :].astype(np.float32)
            self.torque = self.data['motor_torque'][slices, :].astype(np.float32)
        else:
            self.error = self.data['angle_error'].astype(np.float32)
            self.velocity = self.data['motor_velocity'].astype(np.float32)
            self.torque = self.data['motor_torque'].astype(np.float32)
        self.X = np.stack((self.error[2 * history_interval:-1, ].flatten(),
                           self.error[history_interval:-1 - history_interval, ].flatten(),
                           self.error[:-1 - 2 * history_interval, ].flatten(),
                           self.velocity[2 * history_interval:-1, ].flatten(),
                           self.velocity[history_interval:-1 - history_interval, ].flatten(),
                           self.velocity[:-1 - 2 * history_interval, ].flatten()), axis=1)
        self.Y = np.expand_dims(self.torque[1 + 2 * history_interval:, ].flatten(), axis=1)
        self.size = len(self.error)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_actuator_net(actuator_net, dataset_class, history_interval=5,
                       lr=1e-3, num_epochs=1000, batch_size=1000, device='cuda'):
    device = torch.device(device)
    dataset_paths = [os.path.join(RSC_DIR, filename) for filename in os.listdir(RSC_DIR)]
    dataset = ConcatDataset([dataset_class(path, history_interval) for path in dataset_paths if path.endswith('npz')])
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_data, test_data = random_split(dataset, (train_len, test_len))
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    net = actuator_net.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    wandb.init(project='actuator_net', name=str((actuator_net.hidden_dims, lr)), mode=None)
    init_logger()
    log_dir = f'ident/{get_timestamp()}'
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
        log_info(f'Epoch {i:>4}/{num_epochs} train loss {train_loss:.6f}')

        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            loss = criterion(net(X), Y)
            test_loss += loss.item()
        test_loss /= test_len
        log_info(f'Epoch {i:>4}/{num_epochs} test loss {test_loss:.6f}')
        wandb.log({'Train/loss': train_loss, 'Test/loss': test_loss})
        if i % 10 == 0:
            torch.save({'model': net.state_dict(), 'hidden_dims': net.hidden_dims},
                       os.path.join(log_dir, f'{i}.pt'))


def get_statistics(model, history_interval=5):
    dataset_paths = [os.path.join(RSC_DIR, filename) for filename in os.listdir(RSC_DIR)]
    dataset = ConcatDataset([DatasetClass(path, history_interval) for path in dataset_paths if path.endswith('npz')])

    criterion = nn.MSELoss(reduction='sum')
    test_loader = DataLoader(dataset, 1000, shuffle=True)
    test_loss = 0.
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        loss = criterion(model(X), Y)
        test_loss += loss.item()
    mse = test_loss / (len(dataset) - 1)
    print('MSE:', mse)
    print('RMSE:', math.sqrt(mse))


if __name__ == '__main__':
    import sys
    import os
    from os.path import dirname, abspath
    import wandb

    sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
    import burl
    from burl.utils import log_info, init_logger
    from burl.exp import get_timestamp

    RSC_DIR = os.path.join(burl.rsc_path, 'motor_data')
    np.set_printoptions(3, linewidth=10000, suppress=True)
    use_history_info = True
    train = False
    DatasetClass = RobotDatasetWithHistory if use_history_info else RobotDataset
    ActuatorNetClass = ActuatorNetWithHistory if use_history_info else ActuatorNet
    device = 'cuda'
    if train:
        actuator_net = ActuatorNetClass(hidden_dims=(32, 32, 32))
        train_actuator_net(actuator_net, DatasetClass, lr=1e-4, num_epochs=2000, device=device)
        get_statistics(actuator_net)
    else:
        # file_name = 'actuator_net_with_history.pt' if use_history_info else 'actuator_net.pt'
        # model_path = os.path.join(burl.rsc_path, file_name)
        model_path = '/home/jewel/Workspaces/teacher-student-main/burl/sim/ident/22-05-04_11-01-40/2000.pt'
        model_info = torch.load(model_path, map_location={'cuda:0': device})
        actuator_net = ActuatorNetClass(hidden_dims=model_info['hidden_dims']).to(device)
        actuator_net.load_state_dict(model_info['model'])
        get_statistics(actuator_net)

        dataset = DatasetClass(os.path.join(RSC_DIR, 'state_cmd_data_NoLoadT0.4.npz'))
        motor_idx = 2
        slices = slice(5000, 5500)
        if not use_history_info:
            error = dataset.error[slices, motor_idx]
            error_rate = dataset.error_rate[slices, motor_idx]
            velocity = dataset.velocity[slices, motor_idx]
            torque = dataset.torque[slices, motor_idx]
            predicted = actuator_net.calc_torque(error, error_rate, velocity)
        else:
            error = dataset.error[slices, motor_idx]
            error_his1 = dataset.error[slice(slices.start - 5, slices.stop - 5), motor_idx]
            error_his2 = dataset.error[slice(slices.start - 10, slices.stop - 10), motor_idx]
            velocity = dataset.velocity[slices, motor_idx]
            velocity_his1 = dataset.velocity[slice(slices.start - 5, slices.stop - 5), motor_idx]
            velocity_his2 = dataset.velocity[slice(slices.start - 10, slices.stop - 10), motor_idx]
            error_rate = dataset.data['angle_error_rate'][slices, motor_idx].astype(np.float32)
            torque = dataset.torque[slices.start + 1:slices.stop + 1, motor_idx]
            predicted = actuator_net.calc_torque(error, error_his1, error_his2, velocity, velocity_his1, velocity_his2)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(9.6, 6.4), gridspec_kw={'height_ratios': [1, 1, 2]})
        ax[0].plot(error)
        ax[0].set_ylabel('error(rad)', rotation=0)
        ax[0].yaxis.set_label_coords(-0.05, 0.99)
        ax[0].set_xticks([])
        ax[1].plot(velocity)
        ax[1].set_xticks([])
        ax[1].set_ylabel('velocity(rad$\cdot$s$^{-1}$)', rotation=0)
        ax[1].yaxis.set_label_coords(-0.03, 1.03)
        ax[2].set_xlabel('sample point number')
        ax[2].set_ylabel('torque')
        ax[2].set_ylabel('torque(N$\cdot$m)', rotation=0)
        ax[2].yaxis.set_label_coords(-0.03, 0.99)
        ax[2].plot(torque, linewidth=1)
        ax[2].plot(predicted, linewidth=1, linestyle='--')
        ax[2].plot(error * 150 - velocity * 4, linewidth=1, linestyle='dashdot')
        ax[2].set_ylim(-20, 25)
        ax[2].legend(['measured', 'identified', 'vanilla pd'])
        fig.show()
        fig.savefig('acnet.pdf')
