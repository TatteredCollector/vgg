import torch
from torchvision import transforms, datasets
from torch import utils
from model.vgg import vgg
from torch import nn, optim
from tqdm import  tqdm

import os
import json


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                          std=(0.5, 0.5, 0.5))
                                     ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   # resize需要规定 w 和 h 两个参数
                                   # 只规定一个，值缩放规定的
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                        std=(0.5, 0.5, 0.5))])

    }
    # os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # 返回绝对路径 可以对路径操作 上例是返回上两级绝对目录
    # print(data_root)
    data_root = os.getcwd()
    img_path = os.path.join(data_root, 'data')
    assert os.path.exists(img_path), "{} path is not exist!".format(img_path)
    train_data = datasets.ImageFolder(root=os.path.join(img_path, 'train'),
                                      transform=data_transform["train"])
    classes_dict = train_data.class_to_idx
    classes_dict = dict((key, val) for val, key in classes_dict.items())
    # dump：将dict类型转换为json字符串格式，写入到文件 （易存储）
    # dumps： 是将dict转换为string
    Josn_dict = json.dumps(classes_dict, indent=4)
    with open("class_dict.json", 'w') as f:
        f.write(Josn_dict)

    batch_size = 32
    nm = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("using number workers is {}.".format(nm))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nm)
    # tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
    #                                batch_size=2,     # 输出的batch size
    #                                shuffle=True,     # 随机输出
    #                                num_workers=0)    # 只有1个进程

    val_data = datasets.ImageFolder(root=os.path.join(img_path, 'val'),
                                    transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=nm)
    num_train = len(train_data)
    num_val = len(val_data)
    print("using {} images train,using {} images val.".format(num_train, num_val))

    model_name = 'vgg16'
    net = vgg(model_name, num_classes=5, init_weight=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = "./{}Net.pth".format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        all_loss = 0.0
        train_bar = tqdm(train_loader,)
        for steps, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            all_loss += loss
            train_bar.desc = "train epoch:{}/{}, loss : {:.3f}".format(epoch,
                                                                       epochs,
                                                                       loss)
        #val
        net.eval()
        acc = 0.0
        # with 语句实质是上下文管理。
        # 1、上下文管理协议。包含方法__enter__() 和 __exit__()，支持该协议对象要实现这两个方法。
        # 2、上下文管理器，定义执行with语句时要建立的运行时上下文，负责执行with语句块上下文中的进入与退出操作。
        # 3、进入上下文的时候执行__enter__方法，如果设置as var语句，var变量接受__enter__()方法返回值。
        # 4、如果运行时发生了异常，就退出上下文管理器。调用管理器__exit__方法。
        # 它将常用的 try ... except ... finally ... 模式很方便的被复用。
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for data in val_bar:
                val_images, val_label = data
                outputs = net(val_images.to(device))
                predict = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict, val_label.to(device)).sum().item()
        val_acc = acc / num_val

        print("epoch : {} train_loss: {:.3f}  val_acc: {:.3f}".format(epoch+1,
                                                                      all_loss/train_steps,
                                                                      val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
    print("finished training ")


if __name__ == "__main__":
    train()
