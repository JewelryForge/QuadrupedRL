from typing import Union

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size].contiguous()


class CausalConv(nn.Module):
    """input: ... old ... new"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: Union[str, int] = 0, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 0, dilation)
        if padding != 0 and padding != 'valid':
            self.padding = nn.ConstantPad1d((padding, 0), 0.)
            self.conv = nn.Sequential(self.conv, self.padding)

    def forward(self, x):
        return self.conv(x)


class Add(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.transforms = args

    def forward(self, x):
        return torch.add(*[trans(x) for trans in self.transforms])


class TCNEncoder(nn.Module):
    input_dim = (60, 100)

    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv(60, 38, 5, stride=1, padding=4, dilation=1)
        self.skip_conv1 = nn.Conv1d(60, 38, 1, stride=1, padding=0, dilation=1)
        self.relu = nn.ReLU()
        self.dsp1 = nn.Conv1d(38, 38, 3, stride=2, padding=1, dilation=1)
        self.conv2 = CausalConv(38, 38, 5, stride=1, padding=8, dilation=2)
        self.skip_conv2 = nn.Conv1d(38, 38, 1, stride=1, padding=0, dilation=1)
        self.dsp2 = nn.Conv1d(38, 38, 3, stride=2, padding=1, dilation=1)
        self.conv3 = CausalConv(38, 38, 5, stride=1, padding=16, dilation=4)
        self.skip_conv3 = nn.Conv1d(38, 38, 1, stride=1, padding=0, dilation=1)
        self.dsp3 = nn.Conv1d(38, 38, 3, stride=2, padding=1, dilation=1)
        self.linear = nn.Linear(494, 64)
        self.layers = nn.Sequential(
            Add(self.conv1, self.skip_conv1), self.relu, self.dsp1,
            Add(self.conv2, self.skip_conv2), self.relu, self.dsp2,
            Add(self.conv3, self.skip_conv3), self.relu, self.dsp3,
            nn.Flatten(), self.linear
        )

    def forward(self, x):
        return self.layers(x)


class TCNEncoderNoPadding(nn.Module):
    input_dim = (60, 123)

    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv(60, 38, 5, stride=1, padding=0, dilation=1)
        self.skip_conv1 = nn.Sequential(Chomp1d(4), nn.Conv1d(60, 38, 1, stride=1, padding=0, dilation=1))
        self.relu = nn.ReLU()
        self.dsp1 = nn.Conv1d(38, 38, 3, stride=2, padding=0, dilation=1)
        self.conv2 = CausalConv(38, 38, 5, stride=1, padding=0, dilation=2)
        self.skip_conv2 = nn.Sequential(Chomp1d(8), nn.Conv1d(38, 38, 1, stride=1, padding=0, dilation=1))
        self.dsp2 = nn.Conv1d(38, 38, 3, stride=2, padding=0, dilation=1)
        self.conv3 = CausalConv(38, 38, 5, stride=1, padding=0, dilation=4)
        self.skip_conv3 = nn.Sequential(Chomp1d(16), nn.Conv1d(38, 38, 1, stride=1, padding=0, dilation=1))
        self.dsp3 = nn.Conv1d(38, 38, 3, stride=2, padding=0, dilation=1)
        self.linear = nn.Linear(4 * 38, 64)  # history len 123
        self.layers = nn.Sequential(
            Add(self.conv1, self.skip_conv1), self.relu, self.dsp1,
            Add(self.conv2, self.skip_conv2), self.relu, self.dsp2,
            Add(self.conv3, self.skip_conv3), self.relu, self.dsp3,
            nn.Flatten(), self.linear
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    net = TCNEncoder()
    import torchsummary

    torchsummary.summary(net, (60, 100), device='cpu')
    # for i in range(100, 150):
    #     feature = torch.randn(2, 60, i)
    #     net = TCNEncoderNoPadding()
    #     print(i, net(feature).shape)
