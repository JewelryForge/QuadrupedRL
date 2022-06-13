import argparse
import math
import os

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

from qdpgym.sim.common.identify import ActuatorNet
from qdpgym.utils import get_timestamp, log


class RobotDatasetWithHistory(Dataset):
    def __init__(self, path, history_interval, slices=None):
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


def train_actuator_net(actuator_net, dataset_class, history_interval,
                       lr, num_epochs, batch_size, device):
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
        log.info(f'Epoch {i:>4}/{num_epochs} train loss {train_loss:.6f}')

        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            loss = criterion(net(X), Y)
            test_loss += loss.item()
        test_loss /= test_len
        log.info(f'Epoch {i:>4}/{num_epochs} test loss {test_loss:.6f}')
        wandb.log({'Train/loss': train_loss, 'Test/loss': test_loss})
        if i % 10 == 0:
            torch.save({'model': net.state_dict(), 'hidden_dims': net.hidden_dims},
                       os.path.join(log_dir, f'{i}.pt'))


def get_statistics(model, history_interval):
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
    parser = argparse.ArgumentParser(description='Train or test quadruped locomotion.')
    parser.add_argument('-r', '--rsc', metavar='</rsc-dir>',
                        help='motor data directory', required=True)
    parser.add_argument('-i', '--interval', type=int, help='history interval', default=3)
    parser.add_argument('-b', '--batch-size', type=int, help='batch size', default=1000)
    parser.add_argument('-l', '--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('-e', '--epoch', type=int, help='num epochs', default=2000)
    parser.add_argument('--cuda', action='store_true', help='use cuda for training')
    parser.add_argument('--train', action='store_true', help='if train else test')
    args = parser.parse_args()

    RSC_DIR = args.rsc
    if args.cuda or torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    TRAIN = args.train
    HISTORY_INTERVAL = args.interval
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.epoch

    np.set_printoptions(3, linewidth=10000, suppress=True)
    DatasetClass = RobotDatasetWithHistory
    device = torch.device(DEVICE)
    if TRAIN:
        actuator_net = ActuatorNet(hidden_dims=(32, 32, 32))
        train_actuator_net(
            actuator_net,
            DatasetClass,
            history_interval=HISTORY_INTERVAL,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            device=device
        )
        get_statistics(actuator_net, HISTORY_INTERVAL)
    else:
        model_path = '/home/jewel/Workspaces/QuadrupedRLv2/qdpgym/sim/resources/acnet_220526.pt'
        model_info = torch.load(model_path, map_location=device)
        actuator_net = ActuatorNet(hidden_dims=model_info['hidden_dims']).to(device)
        actuator_net.load_state_dict(model_info['model'])
        get_statistics(actuator_net, HISTORY_INTERVAL)

        # dataset = DatasetClass(os.path.join(RSC_DIR, 'state_cmd_data_NoLoadT0.4.npz'))
        # motor_idx = 2
        # slices = slice(5000, 5500)
        #
        # error = dataset.error[slices, motor_idx]
        # error_his1 = dataset.error[slice(slices.start - 5, slices.stop - 5), motor_idx]
        # error_his2 = dataset.error[slice(slices.start - 10, slices.stop - 10), motor_idx]
        # velocity = dataset.velocity[slices, motor_idx]
        # velocity_his1 = dataset.velocity[slice(slices.start - 5, slices.stop - 5), motor_idx]
        # velocity_his2 = dataset.velocity[slice(slices.start - 10, slices.stop - 10), motor_idx]
        # error_rate = dataset.data['angle_error_rate'][slices, motor_idx].astype(np.float32)
        # torque = dataset.torque[slices.start + 1:slices.stop + 1, motor_idx]
        # predicted = actuator_net.calc_torque(error, error_his1, error_his2, velocity, velocity_his1, velocity_his2)

        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots(3, 1, figsize=(9.6, 6.4), gridspec_kw={'height_ratios': [1, 1, 2]})
        # ax[0].plot(error)
        # ax[0].set_ylabel('error(rad)', rotation=0)
        # ax[0].yaxis.set_label_coords(-0.05, 0.99)
        # ax[0].set_xticks([])
        # ax[1].plot(velocity)
        # ax[1].set_xticks([])
        # ax[1].set_ylabel('velocity(rad$\cdot$s$^{-1}$)', rotation=0)
        # ax[1].yaxis.set_label_coords(-0.03, 1.03)
        # ax[2].set_xlabel('sample point number')
        # ax[2].set_ylabel('torque')
        # ax[2].set_ylabel('torque(N$\cdot$m)', rotation=0)
        # ax[2].yaxis.set_label_coords(-0.03, 0.99)
        # ax[2].plot(torque, linewidth=1)
        # ax[2].plot(predicted, linewidth=1, linestyle='--')
        # ax[2].plot(error * 150 - velocity * 4, linewidth=1, linestyle='dashdot')
        # ax[2].set_ylim(-20, 25)
        # ax[2].legend(['measured', 'identified', 'vanilla pd'])
        # fig.show()
        # fig.savefig('acnet.pdf')
