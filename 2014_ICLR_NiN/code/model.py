import torch
import torch.nn as nn


def nin_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 1),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 1),
                         nn.ReLU(inplace=True)
                         )


class NiN(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(NiN, self).__init__()
        self.classifier = nn.Sequential(
            nin_block(3, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
        )
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NiN().to(device)
    summary(model, (3, 224, 224), device=str(device))
