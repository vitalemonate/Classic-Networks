import torch
import torch.nn as nn

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super(VGG, self).__init__()
        self.feature = features
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.feature(x)
        output = torch.flatten(output, start_dim=1)  # 从第1维开始flatten
        output = self.classifier(output)

        return output


def make_layer(cfg, batch_norm=False):
    layers = []

    inchannels = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(inchannels, l, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            inchannels = l
    return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(make_layer(cfgs['A'], batch_norm=True), num_class=100)


def vgg13_bn():
    return VGG(make_layer(cfgs['B'], batch_norm=True), num_class=100)

def vgg16_bn():
    return VGG(make_layer(cfgs['D'], batch_norm=True), num_class=100)

def vgg19_bn():
    return VGG(make_layer(cfgs['E'], batch_norm=True), num_class=100)

