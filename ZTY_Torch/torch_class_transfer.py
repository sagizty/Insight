"""
版本 4月23日 实现分类任务的迁移学习

Backbone：Resnet50

主要任务：
1构建imagefolder dataloader
2搭建训练框架
3配置tensorboard画图
4画grad cam

任务目标：
测试迁移学习的效果



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




4 major scenarios:

1.New dataset is small and similar to original dataset. 
Since the data is small,it is not a good idea to fine-tune the ConvNet due to overfitting concerns.
Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant
to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

2.New dataset is large and similar to the original dataset. 
Since we have more data, we can have more confidence that we won’t overfit 
if we were to try to fine-tune through the full network.

3.New dataset is small but very different from the original dataset. 
Since the data is small, it is likely best to only train a linear classifier. 
Since the dataset is very different, it might not be best to train the classifier form the top of the network, 
which contains more dataset-specific features.
Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.

4.New dataset is large and very different from the original dataset. 
Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. 
However, in practice it is very often still beneficial to initialize with weights from a pretrained model. 
In this case, we would have enough data and confidence to fine-tune through the entire network.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchsummary import summary
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from tensorboardX import SummaryWriter
from PIL import Image

from torch_7_grad_CAM import get_last_conv_name, GradCAM, gen_cam


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# 数据设置
'''
draw_path = '/home/ZTY/runs/lung_cls2_resnet50'

if os.path.exists(draw_path):
    del_file(draw_path)  # 每次开始的时候都先清空一次
# 在命令行同文件夹跑tensorboard --logdir=/home/ZTY/runs --host=10.201.10.16 --port=7777

model_path = '/home/ZTY/saved_models'
model_path = model_path + '/Resnet50_cam_detction_30.pth'

dataroot = '/home/NSCLC-project/Datasets/2D_local_dataset_cls2_Ori_spacing'

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(400),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),  # 注意，这里的预处理参数是按照预训练的配置来设置的，这样迁移学习才能有更明显的效果
    'val': transforms.Compose([
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ]),
}

lung2d_datasets = {x: datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(lung2d_datasets[x], batch_size=50, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这个是为了之后走双卡

class_names = ['ADC', 'SCC']  # A G

dataset_sizes = {x: len(lung2d_datasets[x]) for x in ['train', 'val']}  # 数据数量

'''
# 原始设置
'''
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
'''


def imshow(inp, title=None):  # 与data_transforms对应
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    '''
    # 因为 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    '''
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 测试一波数据
'''
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
# 开始训练，新增这两个内容
# Scheduling the learning rate
# Saving the best model
'''


# Grad CAM部分


# 训练部分

def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # 迭代过程中选用更好的结果

    '''

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False
    '''
    return True


def train_model(model, dataloaders, criterion, optimizer, scheduler, class_names, dataset_sizes, num_epochs=25,
                check_num=100, device=None, draw_path='/home/ZTY/imaging_results', writer=None):
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

            index = 0
            model_time = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # 记录表现
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # 不同任务段用不同dataloader的数据
                inputs = inputs.to(device)
                # print('inputs[0]',type(inputs[0]))

                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # 要track grad if only in train！包一个 with torch.set_grad_enabled(phase == 'train'):不然就True就行
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # preds是最大值出现的位置，相当于是类别id
                    loss = criterion(outputs, labels)  # loss是基于输出的vector与onehot label做loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计表现总和
                log_running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 记录内容给tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' loss',
                                      float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' ACC',
                                      float(torch.sum(preds == labels.data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # 画图检测效果
                if index % check_num == check_num - 1:
                    model_time = time.time() - model_time

                    epoch_idx = epoch + 1
                    print('Epoch:', epoch_idx, '   ', phase, 'index of 100 minibatch:', index // check_num + 1,
                          '     time used:', model_time)

                    print('loss:', float(log_running_loss) / check_num)

                    check_grad_CAM(model_ft, dataloaders, class_names, num_images=2, device=device,
                                   pic_name='GradCAM_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                   draw_path=draw_path, writer=writer)

                    visualize_check(model, dataloaders, class_names, num_images=9, device=device,
                                    pic_name='Visual_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                    draw_path=draw_path, writer=writer)

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

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

    # 记录内容给tensorboard
    if writer is not None:
        writer.close()

    # load best model weights as final model training result 这也是一种避免过拟合的方法
    model.load_state_dict(best_model_wts)
    return model


def visualize_check(model, dataloaders, class_names, num_images=9, device='cpu', pic_name='test',
                    draw_path='/home/ZTY/imaging_results', writer=None):  # 预测测试
    '''
    对num_images个图片进行检查，每行放3个图

    :param model:输入模型
    :param dataloaders:输入数据dataloader，本文件中是train和val的2个dataloader一起作为一个组输入
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量
    :param device:cpu/gpu
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器

    :return:
    '''
    was_training = model.training
    model.eval()

    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                ax.axis('off')
                ax.set_title('Pred: {} True: {}'.format(class_names[preds[j]], class_names[int(labels[j])]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    picpath = draw_path + '/' + pic_name + '.jpg'
                    if not os.path.exists(draw_path):
                        os.makedirs(draw_path)

                    # plt.savefig(picpath)
                    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                    # plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
                    plt.show()
                    plt.savefig(picpath, dpi=1000)

                    model.train(mode=was_training)

                    if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                        image_PIL = Image.open(picpath)
                        img = np.array(image_PIL)
                        writer.add_image(pic_name, img, 1, dataformats='HWC')

                    return

        model.train(mode=was_training)


def check_grad_CAM(model, dataloaders, class_names, num_images=3, device='cpu', pic_name='test',
                   draw_path='/home/ZTY/imaging_results', writer=None):
    '''
    检查num_images个图片在每个类别上的cam，每行有每个类别的图，行数=num_images，为检查的图片数量

    :param model:输入模型
    :param dataloaders:输入数据dataloader，本文件中是train和val的2个dataloader一起作为一个组输入
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量
    :param device:cpu/gpu
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器

    :return:
    '''
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['val']))

    # 预测测试
    was_training = model.training
    model.eval()

    inputs = inputs.to(device)
    labels = classes.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    # 先确定最后一个卷积层名字
    layer_name = get_last_conv_name(model)
    grad_cam = GradCAM(model, layer_name)  # 生成grad cam调取器，包括注册hook等

    images_so_far = 0
    plt.figure()

    for j in range(inputs.size()[0]):

        for cls_idx in range(len(class_names)):
            images_so_far += 1
            ax = plt.subplot(num_images, len(class_names), images_so_far)
            ax.axis('off')
            ax.set_title('True {} Pred {} CAM on {}'.format(class_names[int(labels[j])], class_names[preds[j]],
                                                            class_names[cls_idx]))
            # 基于输入数据 和希望检查的类id，建立对应的cam mask
            mask = grad_cam(inputs[j], cls_idx)
            # 调取原图
            check_image = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            # 转为叠加图cam，与热力图heatmap保存
            cam, heatmap = gen_cam(check_image, mask)

            plt.imshow(cam)
            plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * len(class_names):
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                plt.show()
                plt.savefig(picpath, dpi=1000)

                grad_cam.remove_handlers()  # 删除注册的hook
                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

    grad_cam.remove_handlers()  # 删除注册的hook
    model.train(mode=was_training)


if __name__ == '__main__':
    num_classes = 2

    import notify
    notify.add_text('进行resnet50迁移学习 ' + str(num_classes) + '分类，过程上传到tensorboard')
    notify.send_log()

    draw_path = '/home/ZTY/runs/lung_cls' + str(num_classes) + '_resnet50'
    model_path = '/home/ZTY/saved_models'
    model_path = model_path + '/Resnet50_cam_detction_cls' + str(num_classes) + '_e45.pth'
    dataroot = '/home/NSCLC-project/Datasets/2D_local_dataset_cls' + str(num_classes) + '_Ori_spacing'

    if os.path.exists(draw_path):
        del_file(draw_path)  # 每次开始的时候都先清空一次
    else:
        os.makedirs(draw_path)
    # 在命令行同文件夹跑tensorboard --logdir=/home/ZTY/runs --host=10.201.10.16 --port=7777

    # 调取tensorboard服务器
    writer = SummaryWriter(draw_path)

    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(400),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),  # 注意，这里的预处理参数是按照预训练的配置来设置的，这样迁移学习才能有更明显的效果
        'val': transforms.Compose([
            transforms.CenterCrop(400),
            transforms.ToTensor()
        ]),
    }

    lung2d_datasets = {x: datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(lung2d_datasets[x], batch_size=50, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    class_names = ['ADC', 'SCC', 'LCC', 'NOS'][0:num_classes]  # A G E B
    dataset_sizes = {x: len(lung2d_datasets[x]) for x in ['train', 'val']}  # 数据数量

    # 完全迁移学习：整个搞过来，换层，然后训练全部参数
    model_ft = models.resnet50(pretrained=True)  # True 是预训练好的Resnet50模型，False是随机初始化参数的模型
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # Decide which device we want to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这个是为了之后走双卡
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)
    model_ft.to(device)

    summary(model_ft, input_size=(3, 400, 400))  # to device 之后安排, 输出模型结构

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00005, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.000001)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00002, momentum=0.8)
    # Every step_size epochs, Decay LR by multipling a factor of gamma
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    # train，路径配置的16号服务器的路径，下一步需要给这个train加grad cam
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, class_names, dataset_sizes,
                           num_epochs=45, check_num=50, device=device, draw_path=draw_path, writer=writer)
    # 保存模型
    torch.save(model_ft.state_dict(), model_path)
