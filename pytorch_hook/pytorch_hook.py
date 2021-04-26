import torch
from torchvision.models import resnet18
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


def viz(module, input, output):
    # 这里input是一个是一个元组，是module的__call__函数的参数：
    #     def __call__(self, *input, **kwargs):
    #         for hook in self._forward_pre_hooks.values():
    #             result = hook(self, input)
    #             if result is not None:
    #                 if not isinstance(result, tuple):
    #                     result = (result,)
    #                 input = result
    #         .......
    # 由于input的大小可能不为1,所以在可视化时只考虑第一个元素，所以x = input[0][0](第二个索引是batch)，而不是input[0]
    x = input[0][0]
    y = output[0]
    print(f"x.size() is {x.size()}")
    print(f"y.size() is {y.size()}")
    # 每行最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(2, min_num, i+1)
        plt.imshow(x[i].cpu())
        plt.subplot(2, min_num, i+1+min_num)
        plt.imshow(y[i].cpu())

    plt.show()


def main():
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet18(pretrained=True).to(device)
    for name, m in model.named_modules():
        # 这里只对卷积层的feature map进行显示
        # if name == "conv1":
        #     # register_forward_pre_hook在module的forward函数前执行
        #     # 对应的hook的signature为：hook(module, input) -> None or modified input
        #     # m.register_forward_pre_hook(viz)
        #
        #     # register_forward_hook在module的forward函数后执行
        #     # 对应的hook的signature为：hook(module, input, output) -> None or modified output
        #     m.register_forward_hook(viz)

        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(viz)
    img = cv2.imread('./tulip0.jpg')
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)


if __name__ == '__main__':
    main()
