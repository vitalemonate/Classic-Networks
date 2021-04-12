# 参考
# https://github.com/pytorch/vision/blob/e8dded4c05ee403633529cef2e09bf94b07f6170/torchvision/models/resnet.py#L144
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test5_resnet/model.py
import torch
import torch.nn as nn


# for ResNet-18, ResNet-34
class BasicBlock(nn.Module):
    expansion = 1

    # **kwargs是为了统一BasicBlock和Bottleneck的接口
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        # 在第一个conv进行下采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 找一下如果共享同一个module会有什么影响
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # shortcut connections是否进行下采样
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu2(x)
        return x


# for ResNet-50, ResNet-101, ResNet-152
class Bottleneck(nn.Module):
    # 这一个block输出的通道数是输入通道数的expansion倍
    expansion = 4

    # 这里的out_channels对应的是这个block中前两个卷积核的输出通道数，这个block最终的输出深度是out_channels * expansion
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # 这个宽度是为了适应兼容ResNext，如果是ResNet的模型，那么这个width就是64
        width = int(out_channels * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 与原文的不同的地方
        # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        # while original implementation places the stride at the first 1x1 convolution(self.conv1)
        # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
        # This variant is also known as ResNet V1.5 and improves accuracy according to
        # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.include_top = include_top
        self.groups = groups
        self.width_per_group = width_per_group
        # 7*7卷积核的个数
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, block_nums[0])
        self.layer2 = self.make_layer(block, 128, block_nums[1], stride=2)
        self.layer3 = self.make_layer(block, 256, block_nums[2], stride=2)
        self.layer4 = self.make_layer(block, 512, block_nums[3], stride=2)

        if self.include_top:
            self.avg_pooloing = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 这里的channel对应的是layer中每一个block前两个卷积核的输出界首数
    def make_layer(self, block, channel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channels = channel * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avg_pooloing(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet152().to(device)
    summary(model, (3, 224, 224), device=str(device))
