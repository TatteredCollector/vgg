from model.vgg import vgg
from torchvision import transforms, datasets
import torch

import os
import json
from PIL import Image
import matplotlib.pyplot as plt


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                               std=(0.5, 0.5, 0.5))])
    data_path = os.path.join(os.getcwd(), 'flowers.jpg')
    assert os.path.exists(data_path), "file : {} is not exist!".format(data_path)
    img = Image.open(data_path)
    plt.imshow(img)
    img = data_transforms(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_dict.json'
    assert os.path.exists(json_path), "file: {} is not exist!".format(json_path)
    # load：针对文件句柄，将json格式的字符转换为dict，
    # 从文件中读取 (将string转换为dict)
    # loads：将string转换为dict (将string转换为dict)
    with open(json_path, "r") as f:
        class_dict = json.load(f)

    model_name = "vgg16"
    net = vgg(model_name, num_classes=5, init_weight=False)
    net.to(device)
    net.eval()

    weigths_path = './vgg16Net.pth'
    assert os.path.exists(weigths_path), "file {} is not exist!".format(weigths_path)
    net.load_state_dict(torch.load(weigths_path))

    with torch.no_grad():
        output = torch.squeeze(net(img.to(device))).cpu()

        pred = torch.softmax(output, dim=0)
        pred_cla = torch.argmax(pred, dim=0).numpy()

    context = "class {} prob: {:.3}".format(class_dict[str(pred_cla)],
                                            pred[pred_cla].numpy())
    plt.title(context)

    for i in range(len(pred)):
        print("class: {} prob: {:.3}".format(class_dict[str(i)],
                                             pred[i].numpy()))
    plt.show()


if __name__ == "__main__":
    predict()
