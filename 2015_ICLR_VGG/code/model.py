import torch
import torch.nn as nn


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, num_class=1000, init_weights=False):
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
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        output = self.feature(x)
        output = torch.flatten(output, start_dim=1)  # 从第1维开始flatten
        output = self.classifier(output)

        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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


def vgg11_bn(num_class=1000):
    return VGG(make_layer(cfgs['A'], batch_norm=True), num_class=num_class)


def vgg13_bn(num_class=1000):
    return VGG(make_layer(cfgs['B'], batch_norm=True), num_class=num_class)


def vgg16_bn(num_class=1000):
    return VGG(make_layer(cfgs['D'], batch_norm=True), num_class=num_class)


def vgg19_bn(num_class=1000):
    return VGG(make_layer(cfgs['E'], batch_norm=True), num_class=num_class)


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16_bn().to(device)
    summary(model, (3, 224, 224), device=str(device))


