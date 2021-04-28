import torch
import torch.nn as nn


def make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


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
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1))

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


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()

        input_channels = make_divisible(32*alpha, round_nearest)
        last_channels = make_divisible(1280*alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        features = []
        features.append(ConvBNReLU(3, input_channels, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channels = make_divisible(c*alpha, round_nearest)
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(input_channels, output_channels, stride=s, expand_ratio=t))
                else:
                    features.append(InvertedResidual(input_channels, output_channels, stride=1, expand_ratio=t))
                input_channels = output_channels
        features.append(ConvBNReLU(input_channels, last_channels, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2().to(device)
    summary(model, (3, 224, 224), device=str(device))


















