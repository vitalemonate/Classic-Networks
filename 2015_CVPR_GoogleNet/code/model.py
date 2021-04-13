# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class GoogLeNet(nn.Module):
#     def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
#         super(GoogLeNet, self).__init__()
#
#         self.aux_logits = aux_logits
#
#         self.conv1 = BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
#         self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#
#         self.conv2 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=1)
#         self.conv3 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
#         self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#
#         self.inception3a = Inception_v1(in_channels=192, ch1x1=64, in_ch3x3=96,
#                                         ch3x3=128, in_ch5x5=16, ch5x5=32, pool_proj=32)
#         self.inception3b = Inception_v1(in_channels=256, ch1x1=128, in_ch3x3=128,
#                                         ch3x3=192, in_ch5x5=32, ch5x5=96, pool_proj=64)
#         self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#
#         self.inception4a = Inception_v1(in_channels=480, ch1x1=192, in_ch3x3=96,
#                                         ch3x3=208, in_ch5x5=16, ch5x5=48, pool_proj=64)
#         self.inception4b = Inception_v1(in_channels=512, ch1x1=160, in_ch3x3=112,
#                                         ch3x3=224, in_ch5x5=24, ch5x5=64, pool_proj=64)
#         self.inception4c = Inception_v1(in_channels=512, ch1x1=128, in_ch3x3=128,
#                                         ch3x3=256, in_ch5x5=24, ch5x5=64, pool_proj=64)
#         self.inception4d = Inception_v1(in_channels=512, ch1x1=112, in_ch3x3=144,
#                                         ch3x3=288, in_ch5x5=32, ch5x5=64, pool_proj=64)
#         self.inception4e = Inception_v1(in_channels=528, ch1x1=256, in_ch3x3=160,
#                                         ch3x3=320, in_ch5x5=32, ch5x5=128, pool_proj=128)
#         self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#
#         self.inception5a = Inception_v1(in_channels=832, ch1x1=256, in_ch3x3=160,
#                                         ch3x3=320, in_ch5x5=32, ch5x5=128, pool_proj=128)
#         self.inception5b = Inception_v1(in_channels=832, ch1x1=384, in_ch3x3=192,
#                                         ch3x3=384, in_ch5x5=48, ch5x5=128, pool_proj=128)
#         if aux_logits:
#             self.aux1 = InceptionAux(512, num_classes)
#             self.aux2 = InceptionAux(528, num_classes)
#
#         self.avepool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_classes)
#         if init_weights:
#             self.initialize_weights()
#
#     def forward(self, x):
#         x = self.max_pool1(self.conv1(x))
#         x = self.max_pool2(self.conv3(self.conv2(x)))
#         x = self.max_pool3(self.inception3b(self.inception3a(x)))
#         x = self.inception4a(x)
#         if self.training and self.aux_logits:
#             aux1 = self.aux1(x)
#
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#         if self.training and self.aux_logits:
#             aux2 = self.aux2(x)
#
#         x = self.max_pool4(self.inception4e(x))
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#
#         x = self.avepool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         if self.training and self.aux_logits:
#             return x, aux2, aux1
#
#         return x
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#
# class Inception_v1(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  ch1x1,
#                  in_ch3x3,
#                  ch3x3: int,
#                  in_ch5x5: int,
#                  ch5x5,
#                  pool_proj
#                  ):
#         super(Inception_v1, self).__init__()
#
#         self.branch_1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
#         self.branch_2 = nn.Sequential(BasicConv2d(in_channels, in_ch3x3, kernel_size=1),
#                                       BasicConv2d(in_ch3x3, ch3x3, kernel_size=3, padding=1))
#         self.branch_3 = nn.Sequential(BasicConv2d(in_channels, in_ch5x5, kernel_size=1),
#                                       BasicConv2d(in_ch5x5, ch5x5, kernel_size=5, padding=2))
#         self.branch_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#                                       BasicConv2d(in_channels, pool_proj, kernel_size=1))
#
#     def forward(self, x):
#         branch1 = self.branch_1(x)
#         branch2 = self.branch_2(x)
#         branch3 = self.branch_3(x)
#         branch4 = self.branch_4(x)
#
#         return torch.cat([branch1, branch2, branch3, branch4], dim=1)
#
#
# class InceptionAux(nn.Module):
#     def __init__(self, in_channels, num_classes=1000):
#         super(InceptionAux, self).__init__()
#         self.avg_pooling = nn.AvgPool2d(kernel_size=5, stride=3)
#         self.conv1 = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1)
#         self.fc1 = nn.Linear(16 * 128, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         x = self.avg_pooling(x)
#         x = self.conv1(x)
#         x = torch.flatten(x, 1)
#         x = F.dropout(x, 0.7, training=self.training)
#         x = F.relu(self.fc1(x), inplace=True)
#         x = F.dropout(x, 0.7, training=self.training)
#         x = self.fc2(x)
#         return x
#
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x
#
#
# if __name__ == '__main__':
#     from torchsummary import summary
#
#     device = torch.device("cpu")
#     model = GoogLeNet().to(device)
#     summary(model, (3, 224, 224), device=str(device))

import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
