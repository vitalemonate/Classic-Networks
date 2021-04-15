import torch
import torch.nn as nn


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_dw(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class MobileNet_v1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet_v1, self).__init__()
        self.classifier = nn.Sequential(
            conv_dw(3, 32, stride=2),
            conv_dw(32, 64),
            conv_dw(64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512, stride=2),
            nn.Sequential(*[conv_dw(512, 512) for i in range(5)]),
            conv_dw(512, 1024, stride=2),
            conv_dw(1024, 1024)
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        x = self.avg_pooling(x)
        x = torch.squeeze(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNet_v1().to(device)
    summary(model, (3, 224, 224), device=str(device))
