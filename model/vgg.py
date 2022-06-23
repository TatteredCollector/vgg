from torch import nn
import torch
from torchsummary import summary

"""
vgg 有11 13 16 19layer四种
模块化 均采用3*3大小的卷积，stride = 1 padding = 1
激活函数为Relu
最大池化 2*2 stride = 2 padding=0
最后三个全连接层
输入图像依旧是224*224

初始化方法
Xavier在tanh中表现的很好，但在Relu激活函数中表现的很差，
何凯明提出针对于Relu的初始化方法。

torch.nn里面的
"""


class Classifier(nn.Module):
    def __init__(self, in_feature, classes_num):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_feature, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=classes_num)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def mk_feature(layer_list):
    my_list = []
    in_channel = 3
    for layer in layer_list:
        if layer == "M":
            my_list += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            conv = nn.Conv2d(in_channels=in_channel, out_channels=layer,
                             kernel_size=(3, 3), padding=1)
            my_relu = nn.ReLU(inplace=True)
            my_list += [conv, my_relu]
            in_channel = layer
    '''
    self.features = nn.Module.add_module(self, name="Max_pool"
    module=nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=0)))
    '''
    # 形参——单个星号代表这个位置接收任意多个非关键字参数，转化成元组方式。
    # 实参——如果 * 号加在了是实参上，代表的是将输入迭代器拆成一个个元素。
    # 从nn.Sequential的定义来看，输入要么是orderdict,要么是一系列的模型，
    # 遇到list，必须用*号进行转化，否则会报错
    # TypeError: list is not a Module subclass
    return nn.Sequential(*my_list)


class VGG(nn.Module):
    def __init__(self, feature, in_feature=7 * 7 * 512, num_classes=1000, init_weight=False):
        super(VGG, self).__init__()
        self.feature = feature
        self.classifier = Classifier(in_feature, classes_num=num_classes)
        if init_weight:
            self._init_weight()

    def forward(self, x):
        # 224*224*3 ->7*7*512
        x = self.feature(x)
        x = x.reshape(x.size(0), -1)
        # 7*7*512->25088
        # nn.Flatten是module子类 ，写在初始化函数Sequential里面
        # torch.flatten是函数 写在前向传播中

        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _init_weight(self):
        # nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有module
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果通过创建随机矩阵显式创建权重，则应进行设置mode=‘fan_out’。
                # 如果权重是通过线性层（卷积或全连接）隐性确定的，则需设置mode=fan_in。
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


cfg = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# 1、*args和**kwargs主要用于定义函数的可变参数
#
# 2、*args：发送一个非键值对的可变数量的参数列表给函数
#
# 3、**kwargs：发送一个键值对的可变数量的参数列表给函数


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfg, "Warning: model {} is not exist!".format(model_name)
    my_cfg = cfg[model_name]
    model = VGG(feature=mk_feature(my_cfg), **kwargs)
    return model


if __name__ == "__main__":
    net = vgg()
    inputs = torch.randn(8, 3, 224, 224)
    out = net(inputs)
    print(out.shape)
    net.to(torch.device("cuda:0"))
    summary(net, input_size=(3, 224, 224))
