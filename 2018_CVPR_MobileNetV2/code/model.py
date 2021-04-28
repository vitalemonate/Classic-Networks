import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        hidden_channels = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, out_channels, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        return x
