'''
简单改写的3D resnet50 看一下会不会下降loss

ResNet结构
stem模块实现数据标准化输入
4个stage进行feature extraction。每个stage采用了Bottleneck思路构建的Conv Block与Identity Block
此外还有对接的mlp结构实现分类任务。（若只是做feature extraction，可以后面换成其他的对应的需求）


Stage内的block结构：
Conv Block：输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，它的作用是改变网络的维度；
Identity Block：输入维度和输出维度（通道数和size）相同，可以串联，用于加深网络，从而提高表现。

他们都是由通用的Bottleneck block来构造
Bottleneck block由卷积路线+残差路线2个路线构成，数据的通道数由inplane变为midplane，最后再变为outplane。
卷积路线提取不同深度的特征从而实现全局感知。残差路线本质是在Bottleneck block内部的conv之间实现跳接从而减小深度网络的过拟合
'''

import torch
from torch import nn


# 最核心的模块构建器
class Bottleneck_block_constractor(nn.Module):
    """
    Bottleneck Block的各个plane值：
    inplane：输出block的之前的通道数
    midplane：在block中间处理的时候的通道数（这个值是输出维度的1/4）
    outplane = midplane*self.extention：输出的通道维度数

    stride: 本步骤（conv/identity）打算进行处理的步长设置

    downsample：若为conv block，传入将通道数进行变化的conv层，把通道数由inplane卷为outplane
    identity block则没有这个问题，因为该模块的 inplane=outplane
    """

    # 每个stage中维度拓展的倍数：outplane相对midplane的放大倍数
    extention = 4

    # 定义初始化的网络和参数
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck_block_constractor, self).__init__()
        # 计算输出通道维度
        outplane = midplane * self.extention

        # 只在这里操作步长，其余卷积目标是小感受区域的信息
        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)

        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)

        self.conv3 = nn.Conv2d(midplane, outplane, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midplane * self.extention)

        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        # 卷积操作forward pass，标准的卷，标，激过程（cbr）
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # 残差信息是否直连（如果时Identity block就是直连；如果是Conv Block就需要对参差边进行卷积，改变通道数和size使得它和outplane一致）
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            # 参差数据直接传过来
            residual = x

        # 此时通道数一致了，将参差部分和卷积部分相加。
        out += residual

        # 最后再进行激活
        out = self.relu(out)

        # 我的理解：其实绝大部分被激活的信息来自residential路线，这样也因此学习得比较慢可是不容易过拟合

        return out


# 网络构建器
class ResNet(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block_constractor, bottleneck_channels_setting, identity_layers_setting, stage_stride_setting,
                 num_classes=None):
        # self.inplane为当前的fm的通道数
        self.inplane = 64
        self.num_classes = num_classes

        super(ResNet, self).__init__()  # 这个递归写法是为了拿到自己这个class里面的其他函数进来

        # 关于模块结构组的构建器
        self.block_constractor = block_constractor
        # 每个stage中Bottleneck Block的中间维度，输入维度取决于上一层
        self.bcs = bottleneck_channels_setting  # [64, 128, 256, 512]
        # 每个stage的conv block后跟着的identity block个数
        self.ils = identity_layers_setting  # [3, 4, 6, 3]
        # 每个stage的conv block的步长设置
        self.sss = stage_stride_setting  # [1, 2, 2, 2]

        # stem的网络层
        # 将RGB图片的通道数卷为inplane
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # 构建每个stage
        self.layer1 = self.make_stage_layer(self.block_constractor, self.bcs[0], self.ils[0], self.sss[0])
        self.layer2 = self.make_stage_layer(self.block_constractor, self.bcs[1], self.ils[1], self.sss[1])
        self.layer3 = self.make_stage_layer(self.block_constractor, self.bcs[2], self.ils[2], self.sss[2])
        self.layer4 = self.make_stage_layer(self.block_constractor, self.bcs[3], self.ils[3], self.sss[3])

        # 后续的网络
        if self.num_classes is not None:
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * self.block_constractor.extention, num_classes)

    def forward(self, x):
        # 定义构建的模型中的数据传递方法

        # stem部分:conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Resnet block实现4个stage
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.num_classes is not None:
            # 对接mlp来做分类
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)

        return out

    def make_stage_layer(self, block_constractor, midplane, block_num, stride=1):
        """
        block:模块构建器
        midplane：每个模块中间运算的维度，一般等于输出维度/4
        block_num：重复次数
        stride：Conv Block的步长
        """

        block_list = []

        # 先计算要不要加downsample模块
        outplane = midplane * block_constractor.extention  # extention存储在block_constractor里面

        if stride != 1 or self.inplane != outplane:
            # 若步长变了，则需要残差也重新采样。 若输入输出通道不同，残差信息也需要进行对应尺寸变化的卷积
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, outplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block_constractor.extention)
            )  # 注意这里不需要激活，因为我们要保留原始残差信息。后续与conv信息叠加后再激活
        else:
            downsample = None

        # 每个stage都是1个改变采样的 Conv Block 加多个加深网络的 Identity Block 组成的

        # Conv Block
        conv_block = block_constractor(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)

        # 更新网络下一步stage的输入通道要求（同时也是内部Identity Block的输入通道要求）
        self.inplane = outplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block_constractor(self.inplane, midplane, stride=1, downsample=None))

        return nn.Sequential(*block_list)  # pytorch对模块进行堆叠组装后返回


"""
测试
resnet50 = ResNet(block_constractor=Bottleneck_block_constractor,
                  bottleneck_channels_setting=[64, 128, 256, 512],
                  identity_layers_setting=[3, 4, 6, 3],
                  stage_stride_setting=[1, 2, 2, 2],
                  num_classes=1000)

x = torch.randn(1, 3, 224, 224)
x = resnet50(x)
print(x.shape)
"""





