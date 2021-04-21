"""
版本 4。1 17：00

Pytorch迁移学习的资料

简单说一下迁移学习，一般来说有几种做法：
1。迁移某一部分层过来
2。对某一部分层进行冻结，从而使他们保持原参数与作用，改变/新增部分层进行训练，使得网络有新的功能
3。对整个网络在新的数据上训练，迁移学习此时只是获得一个更好的初始化参数


流程资料
详细理论+使用场景的解析(很重要)
http://blog.itpub.net/29829936/viewspace-2641919/

官网
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Pytorch学习(十二)—迁移学习Transfer Learning
https://www.jianshu.com/p/d04c17368922



代码细节

1.保存/载入模型的参考资料
https://blog.csdn.net/strive_for_future/article/details/83240081

2.Pytorch模型迁移和迁移学习,导入部分模型参数(很重要)
 https://blog.csdn.net/lu_linux/article/details/113373016

3.冻结与解冻层(很重要)
https://www.zhihu.com/question/311095447/answer/589307812

4.pytorch中存储各层权重参数时的命名规则
https://blog.csdn.net/u014734886/article/details/106230535
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# 处理数据
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),  # 注意，这里的预处理参数是按照预训练的配置来设置的，这样迁移学习才能有更明显的效果
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


'''
# 测试一波数据

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
'''


# 开始训练，新增这两个内容
# Scheduling the learning rate
# Saving the best model

def better_performance(temp_acc, temp_vac, best_acc, best_vac):
    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_acc + temp_vac * 1.2 >= best_acc + best_vac * 1.2:
        return True
    else:
        return False


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=None):
    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # 用来保存最好的模型参数
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy 防止copy的是内存地址，这里因为目标比较大，用这个保证摘下来

    # 初始化最好的表现
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    epoch_idx = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 采用这个写法来综合写train与val过程
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # 记录表现
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # 不同任务段用不同dataloader的数据
                inputs = inputs.to(device)
                print('inputs[0]',type(inputs[0]))

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # preds是最大值出现的位置，相当于是类别id
                    loss = criterion(outputs, labels)  # loss是基于输出的vector与onehot label做loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计表现总和
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # 记录输出本轮情况
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # 假设这里是train的时候，记得记录

            # deep copy the model，如果本epoch为止比之前都表现更好才刷新参数记录
            # 目的是在epoch很多的时候，表现开始下降了，那么拿中间的就行
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac):  # 需要定义"更好"
                epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch idx: ', epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))

    # load best model weights as final model training result 这也是一种避免过拟合的方法
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):  # 预测测试
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)


"""
ResNet 迁移学习

from torch_myResNet import Bottleneck_block_constractor, ResNet
from torchvision import models

# 载入预训练模型
model = models.resnet50(pretrained=True)

PATH = './saved_model_pretrained_resnet50.pth'

# 保存/载入模型的参考资料 https://blog.csdn.net/strive_for_future/article/details/83240081
# Pytorch模型迁移和迁移学习,导入部分模型参数 https://blog.csdn.net/lu_linux/article/details/113373016
# pytorch中存储各层权重参数时的命名规则 https://blog.csdn.net/u014734886/article/details/106230535/

# 保存模型
torch.save(model.state_dict(), PATH)

# 载入保存的模型，首先定义一个空模型用来接住预训练的参数，此时网络中每个层名字要和预训练的命名一致
my_model = ResNet(block_constractor=Bottleneck_block_constractor,
                  bottleneck_channels_setting=[64, 128, 256, 512],
                  identity_layers_setting=[3, 4, 6, 3],
                  stage_stride_setting=[1, 2, 2, 2],
                  num_classes=1000)
my_model.load_state_dict(torch.load(PATH))  # 有的时候会报错，因为层内每个位置的名字设置不一样/结构不匹配

# 使用strict参数，如果为True，表明预训练模型的层和自己定义的网络结构层严格对应相等（比如层名和维度），默认 strict=True
# my_model.load_state_dict(torch.load(PATH), strict=False)  # 这里选择为False，则不完全对等，会自动舍去多余的层和其参数。

# 后续可以自己搭建其他网络，然后从这个倒入了参数的模型中拆层过去
"""


# 3种迁移学习的学习：
'''
# 完全迁移学习：整个搞过来，换层，然后训练全部参数
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

# check
visualize_model(model_ft)
'''
# 完全迁移学习：整个搞过来，换层，然后训练全部参数
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train
model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device=device)

# check
visualize_model(model_ft)
'''
# 部分迁移学习：保留feature extractor
model_feature_ex = models.resnet50(pretrained=True)

# 锁住所有参数
for param in model_feature_ex.parameters():
    param.requires_grad = False

# 换层
num_ftrs = model_feature_ex.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_feature_ex.fc = nn.Linear(num_ftrs, 2)

model_feature_ex = model_feature_ex.to(device)

criterion = nn.CrossEntropyLoss()

# 与之前不同，只需要对被换的层的参数进行优化
optimizer_conv = optim.SGD(model_feature_ex.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# train，因为设计迭代改变的部分少了，也会快一些
model_ft = train_model(model_feature_ex, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

# check
visualize_model(model_ft)
'''

'''
# 加层做部分迁移学习：保留feature extractor的部分
model = models.resnet50(pretrained=True)

PATH = './saved_model_pretrained_resnet50.pth'
torch.save(model.state_dict(), PATH)  # 保存下来练习倒入参数

# 锁住所有参数
for param in model.parameters():
    param.requires_grad = False

# 定义另一个模型（需要追加的块）
model2 = nn.Sequential(nn.Linear(1000, 2), nn.ReLU())  # 用sequential搭建线性模型


class adapter(nn.Module):  # 构建一个链接转换器，把2个需要对接的模型作为新模型的2个模块，定义数据传递方式
    def __init__(self, model, newmodel):
        super(adapter, self).__init__()
        self.model1 = model
        self.model2 = newmodel

    def forward(self, x):
        out = self.model1(x)
        out = self.model2(out)
        return out


# newmodel = adapter(model, model2)  # 如果相连设计比较复杂，需要构建这样的一个外包适配器

newmodel = nn.Sequential(model, model2)  # 如果是线性相连，那直接用nn.Sequential就可以实现

newmodel.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, newmodel.parameters()), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# train
newmodel = train_model(newmodel, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

# check
visualize_model(newmodel)
'''

# 迁移学习的模型参数导入问题：

model = models.resnet50(pretrained=True)

PATH = './saved_model_pretrained_resnet50.pth'
torch.save(model.state_dict(), PATH)  # 保存下来练习倒入参数

# 普通的倒入参数
model.load_state_dict(torch.load(PATH))  # 此时完全匹配，不会有问题
# 锁住所有参数
for param in model.parameters():
    param.requires_grad = False

# 当我们希望把预训练的部分倒入一个我们构建好的部分里面时，需要匹配每层的各个参数名字，这样的话非常复杂
# 因为层内每个位置的名字设置不一样/结构不匹配，会报错

# 解决方法1：用链接的方法，把预训练好的带着参数的模型直接用链接的方式构建到新的模型里面

# 定义另一个模型（需要追加的块）
model2 = nn.Sequential(nn.Linear(1000, 2), nn.ReLU())  # 用sequential搭建线性模型
newmodel = nn.Sequential(model, model2)  # 如果是线性相连，那直接用nn.Sequential就可以实现


# 测试与训练
# newmodel.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, newmodel.parameters()), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# newmodel = train_model(newmodel, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
# visualize_model(newmodel)


# 解决方法2：修改预训练好的模型每层参数的名字，之后再用软匹配导入这些层。


def modify_model(pretrained_model_state_dict_path, newmodel, old_prefix, new_prefix):
    """
    本函数用来修改模型的state dict的参数名字，从而为软倒入准备层

    :param pretrained_model_state_dict_path:  预训练state dict的路径
    :param newmodel:  需要适配的新模型
    :param old_prefix:  人工指定需要修改的名字 字段集合
    :param new_prefix:  人工指定需要改成的名字 字段集合

    :return:  返回修改好的模型
    """
    pretrained_dict = torch.load(pretrained_model_state_dict_path)

    model_dict = newmodel.state_dict()

    state_dict = modify_state_dict(pretrained_dict, model_dict, old_prefix, new_prefix)

    newmodel.load_state_dict(state_dict, strict=False)

    return newmodel


def modify_state_dict(pretrained_dict, model_dict, old_prefix, new_prefix):
    """
    修改model 的 state dict

    :param pretrained_dict:  输入的带参数的state dict
    :param model_dict:  需要适配的模型的state dict
    :param old_prefix:  人工指定需要修改的名字 字段集合
    :param new_prefix:  人工指定需要改成的名字 字段集合

    :return: state_dict
    """
    state_dict = {}
    ii = 0

    for k, v in pretrained_dict.items():

        if k in model_dict.keys():  # 输入不需要改的部分state_dict保持不改
            # state_dict.setdefault(k, v)
            state_dict[k] = v

        else:  # 需要修改的地方
            for o, n in zip(old_prefix, new_prefix):
                # 如果o出现在k内部（说明需要修改），把k中间o这个部分改为n
                kk = k.replace(o, n)

                if kk != k:  # 仅取修改的部分作为增量保留到state_dict里面
                    state_dict[kk] = v
                    ii += 1

    print("modified", ii, "places")
    return state_dict


'''
# 假设我们希望把resnet参数导入刚刚做的newmodel里面,对应层需要修改名字
old_prefix = ["layer1", "layer2", "layer3", "layer4"]
new_prefix = ["0.layer1", "0.layer2", "0.layer3", "0.layer4"]

newmodel = modify_model(PATH, newmodel, old_prefix=old_prefix, new_prefix=new_prefix)

# 当然，上述情况比较特殊，只是因为在新模型中，原resnet被作为一个模块所以外包了一个层0。对于大多数情况是需要去做匹配的。

newmodel.to(device)

criterion = nn.CrossEntropyLoss()

# 假设我们想要4个stage（layer1-4）的数值不变，而其他层都需要重新训练
optimizer = optim.SGD([{'params': [param for name, param in newmodel.named_parameters() if 'layer' not in name]}],
                      lr=0.001, momentum=0.9)
# optimizer.add_param_group({'params': newmodel.parameters()})

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

newmodel = train_model(newmodel, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

visualize_model(newmodel)
'''


# 现在测试deep feature的思路，我们需要在fc层引入deep feature，所以fc层必须修改

# 但是，因为引入deep feature涉及对数据流程的变化（deep feature需要在fc层被concat）因此需要重新写模型然后导入预训练参数


# 调取一个预训练带参数的模型
model_feature_ex = models.resnet50(pretrained=True)

'''
# 本段为反面教材

num_ftrs = model_feature_ex.fc.in_features
new_places = 1024
model_feature_ex.fc = nn.Linear(num_ftrs + new_places, 1000)  # 这样是不行的，因为数据流程forward并不匹配！！！！！！！！！1

# 定义另一个模型（需要追加的块）
model2 = nn.Sequential(nn.Linear(1000, 2), nn.ReLU())  # 用sequential搭建线性模型

newmodel = nn.Sequential(model_feature_ex, model2)  # 如果是线性相连，那直接用nn.Sequential就可以实现
'''


# 重新写模型，对原ResNet50模型进行修改
from torch_myResNet import Bottleneck_block_constractor  # 自己的另一个文件，用来做resnet的


# 包含deep feature 的 ResNet50 网络构建器
class ResNet_with_deep_feature(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block_constractor, bottleneck_channels_setting, identity_layers_setting, stage_stride_setting,
                 num_classes=None):
        # self.inplane为当前的fm的通道数
        self.inplane = 64
        self.num_classes = num_classes

        super(ResNet_with_deep_feature, self).__init__()  # 这个递归写法是为了拿到自己这个class里面的其他函数进来

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

            # 换掉的层，故意改个层名，免得被导入数据的时候发现格式不同报错！！！！！！！！！！！！1
            self.fc_deep = nn.Linear(512 * self.block_constractor.extention + 1024, num_classes)

    def forward(self, x, y):  # 这个时候，网络的input=【x，y】x为2d的feature，y为1024维的 机器学习 feature

        # 原 x = torch.randn(1, 3, 224, 224) 现 input = 【torch.randn(1, 3, 224, 224)，torch.randn(1, 1024)】
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
            cnn_feature = torch.flatten(out, 1)
            deep_feature = torch.flatten(y, 1)

            out = torch.cat((cnn_feature, deep_feature), dim=1)

            # 换掉的层，故意改个层名，免得被导入数据的时候发现格式不同报错！！！！！！！！！！！！！！！
            out = self.fc_deep(out)

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


# 测试
resnetDF = ResNet_with_deep_feature(block_constractor=Bottleneck_block_constractor,
                                    bottleneck_channels_setting=[64, 128, 256, 512],
                                    identity_layers_setting=[3, 4, 6, 3],
                                    stage_stride_setting=[1, 2, 2, 2],
                                    num_classes=2)  # 迁移学习到2个类上

# 不能直接改个名就完事，因为层名不同只是表面，本质上数据尺寸不同
# newmodel = modify_model(PATH, resnetDF, old_prefix=['fc'], new_prefix=['fc_deep'])

resnetDF.load_state_dict(torch.load(PATH), strict=False) # 仅fc_deep层有变化。故意改个层名，免得被导入数据的时候发现格式不同报错

# 做25个数据用来测试模型是否对
x=torch.randn(25, 3, 224, 224)
y=torch.randn(25, 1024)

t = resnetDF(x,y)
print(t.shape)


# 假设在训练中，我们先用noise来作为机器学习的feature，现在因为没有dataloader，因此需要改写train和val的代码，让他们自己拿noise数据

def train_mode_with_deepfeaturel(model, criterion, optimizer, scheduler, num_epochs=25, device=None):
    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # 用来保存最好的模型参数
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy 防止copy的是内存地址，这里因为目标比较大，用这个保证摘下来

    # 初始化最好的表现
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    epoch_idx = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 采用这个写法来综合写train与val过程
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # 记录表现
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # 不同任务段用不同dataloader的数据

                noise_holder = torch.randn(inputs.shape[0], 1024).to(device)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, noise_holder)
                    _, preds = torch.max(outputs, 1)  # preds是最大值出现的位置，相当于是类别id
                    loss = criterion(outputs, labels)  # loss是基于输出的vector与onehot label做loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计表现总和
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # 记录输出本轮情况
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # 假设这里是train的时候，记得记录

            # deep copy the model，如果本epoch为止比之前都表现更好才刷新参数记录
            # 目的是在epoch很多的时候，表现开始下降了，那么拿中间的就行
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac):  # 需要定义"更好"
                epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch idx: ', epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))

    # load best model weights as final model training result 这也是一种避免过拟合的方法
    model.load_state_dict(best_model_wts)
    return model


def visualize_model_with_deepfeaturel(model, num_images=6):  # 预测测试
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            noise_holder = torch.randn(inputs.shape[0], 1024).to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, noise_holder)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)

criterion = nn.CrossEntropyLoss()

resnetDF.to(device)

# resnetDF = nn.DataParallel(resnetDF)  # 单机多卡

optimizer = optim.SGD(filter(lambda p: p.requires_grad, newmodel.parameters()), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_mode_with_deepfeaturel(resnetDF, criterion, optimizer, exp_lr_scheduler, num_epochs=25, device=device)

visualize_model_with_deepfeaturel(resnetDF)
