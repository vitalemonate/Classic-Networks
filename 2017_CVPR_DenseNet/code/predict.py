import json

import torch
import os
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from model import densenet121
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # batch predict
    images_path = "../../test_images"
    dirs = os.listdir(images_path)
    image_list = []
    for img_name in dirs:
        img_path = os.path.join(images_path, img_name)
        img = Image.open(img_path)
        img = data_transform(img)
        image_list.append(img)

    batch_image = torch.stack(image_list, dim=0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), f"{json_path} not exists"

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    net = densenet121(num_classes=5, memory_efficient=True).to(device)
    weight_path = "./densenet121.pth"
    net.load_state_dict(torch.load(weight_path, map_location=device))

    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(batch_image.to(device)).cpu())
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        for idx, (pro, cla) in enumerate(zip(probs, classes)):
            print("image: {}  class: {}  prob: {:.3}".format(dirs[idx],
                                                             class_indict[str(cla.numpy())],
                                                             pro.numpy()))
        # visualisation
        plot_show = torchvision.utils.make_grid(batch_image)
        imshow(plot_show, title=[class_indict[str(x.numpy())] for x in classes])


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    main()
