import torch
import torch.nn as nn


def nin_block(in_channel, out_channel, kernel_size, stride, padding=0):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channel, out_channel, 1),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channel, out_channel, 1),
                         nn.ReLU(inplace=True)
                         )


class NiN_model(nn.Module):
    def __init__(self, in_channel):
        super(NiN_model, self).__init__()
        self.cnn_1 = nin_block(in_channel=in_channel, out_channel=96, kernel_size=11, stride=4, padding=0)
        self.polling_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop_out = nn.Dropout()
        self.cnn_2 = nin_block(in_channel=96, out_channel=256, kernel_size=5, stride=1, padding=2)
        self.polling_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.cnn_3 = nin_block(in_channel=256, out_channel=384, kernel_size=3, stride=1, padding=1)
        self.polling_3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.cnn_4 = nin_block(in_channel=384, out_channel=10, kernel_size=3, stride=1, padding=1)
        self.global_average_polling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, X):
        X = self.polling_1(self.cnn_1(X))
        X = self.drop_out(X)
        X = self.polling_2(self.cnn_2(X))
        X = self.drop_out(X)
        X = self.polling_3(self.cnn_3(X))
        X = self.drop_out(X)
        X = self.global_average_polling(self.cnn_4(X))
        X = X.view(X.shape[0], -1)
        return X


if __name__ =='__main__':
    net = NiN_model(in_channel=3)
    X = torch.rand(128, 3, 224, 224)
    # for name, blk in net.named_children():
    #     X = blk(X)
    #     print(name, 'output shape: ', X.shape)
    print(net(X).shape)
