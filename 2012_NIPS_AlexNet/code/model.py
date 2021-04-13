import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),      # input: [3, 224, 224] output: [96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # input: [96, 55, 55] output: [96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),     # input: [96, 27, 27] output: [256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # input: [256, 27, 27] output: [256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),    # input: [256, 13, 13] output: [384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),    # input: [384, 13, 13] output: [384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),    # input: [384, 13, 13] output: [256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                       # input: [256, 13, 13] output: [256, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
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


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    summary(model, (3, 224, 224), device=str(device))
