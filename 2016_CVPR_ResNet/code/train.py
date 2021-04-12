# 参考
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test5_resnet/train.py
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import json
import os
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
# from torchvision.models import resnet34
from tqdm import tqdm
from model import resnet34

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    # 使用ImageNet的均值和标准差
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    image_path = os.path.join(data_root, "data_set", "flower_data")
    assert os.path.exists(image_path), f"{image_path} does not exist"

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0])
    print(f"using {nw} workers to load data")

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    train_num = len(train_dataset)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    val_num = len(val_dataset)

    # 加载数据更精练的一种写法(出自pytorch官方的transfer learning教程)
    # image_datasets = {x: datasets.ImageFolder(os.path.join(image_path, x),
    #                                           data_transform[x])
    #                   for x in ['train', 'val']}
    # dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
    #                                               shuffle=True, num_workers=4)
    #                for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"using {train_num} images for training and {val_num} images for validation.")

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    net = resnet34(num_classes=5)
    # model_weights_path = "./resnet34-pre.pth"
    # assert os.path.exists(model_weights_path), f"pretrained weights {model_weights_path} not exists."
    #
    # net.load_state_dict(torch.load(model_weights_path, map_location=device))
    # # for param in net.parameters():
    # #     param.requires_grad = False
    #
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params,  lr=0.0001)

    epochs = 50
    best_acc = 0.0
    save_path = "./resNet_34.pth"
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()

        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        net.eval()
        acc = 0.0
        val_bar = tqdm(val_loader)
        with torch.no_grad():
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "val epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc/val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finish Training")


if __name__ == "__main__":
    main()
