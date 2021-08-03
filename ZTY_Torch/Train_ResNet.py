"""
版本 6月23日 Resnet50 baseline

Backbone：Resnet50 实现分类任务的迁移学习

数据集:
image folder 格式就行


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
from torchsummary import summary
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from grad_CAM import get_last_conv_name, GradCAM, gen_cam


def setup_seed(seed):  # 设置随机数种子
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


# 获取模型
def get_model(num_classes=1000, edge_size=224, model_idx=None):
    # 完全迁移学习：整个搞过来，换层，然后训练全部参数

    if model_idx[0:8] == 'ResNet34':
        model_ft = models.resnet34(pretrained=True)  # True 是预训练好的Resnet50模型，False是随机初始化参数的模型
    elif model_idx[0:8] == 'ResNet50':
        model_ft = models.resnet50(pretrained=True)  # True 是预训练好的Resnet50模型，False是随机初始化参数的模型
    elif model_idx[0:9] == 'ResNet101':
        model_ft = models.resnet101(pretrained=True)  # True 是预训练好的Resnet50模型，False是随机初始化参数的模型
    else:
        print('this model is not defined in get model')
        return -1

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


# Grad CAM部分
def check_grad_CAM(model, dataloader, class_names, check_index=1, num_images=3, device='cpu', skip_batch=10,
                   pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):
    '''
    检查num_images个图片在每个类别上的cam，每行有每个类别的图，行数=num_images，为检查的图片数量

    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
    :param device:cpu/gpu
    :param pic_name:输出图片的名字
    :param draw_path:输出图片的文件夹
    :param writer:输出图片上传到tensorboard服务器

    :return:
    '''
    # Get a batch of training data
    dataloader = iter(dataloader)
    for i in range(check_index * skip_batch):
        inputs, classes = next(dataloader)

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
            ax.set_title('GT:{} Pr:{} CAM on {}'.format(class_names[int(labels[j])], class_names[preds[j]],
                                                        class_names[cls_idx]))
            # 基于输入数据 和希望检查的类id，建立对应的cam mask
            mask = grad_cam(inputs[j], cls_idx)
            # 调取原图
            check_image = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            # 转为叠加图cam，与热力图heatmap保存
            cam, heatmap = gen_cam(check_image, mask)

            plt.imshow(cam)  # 接收一张图像，只是画出该图，并不会立刻显示出来。
            plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * len(class_names):
                picpath = draw_path + '/' + pic_name + '.jpg'
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                '''
                myfig = plt.gcf()  # get current image
                myfig.savefig(picpath, dpi=1000)
                '''
                plt.savefig(picpath, dpi=1000)
                plt.show()

                grad_cam.remove_handlers()  # 删除注册的hook
                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

    grad_cam.remove_handlers()  # 删除注册的hook
    model.train(mode=was_training)


# 训练部分

def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # 迭代过程中选用更好的结果

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, num_epochs=25,
                check_minibatch=100, scheduler=None, device=None, draw_path='/home/ZTY/imaging_results',
                enable_attention_check=False, enable_visualize_check=False, writer=None):
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
    best_epoch_idx = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 采用这个写法来综合写train与val过程

            index = 0
            model_time = time.time()

            # 初始化计数字典
            log_dict = {}
            for cls_idx in range(len(class_names)):
                log_dict[cls_idx] = {'tp': 0, 'tp_plus_fp': 0, 'tp_plus_fn': 0}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # 初始化记录表现
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0
            check_dataloaders = copy.deepcopy(dataloaders)

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

                # Compute recision and recall for each class.
                for cls_idx in range(len(class_names)):
                    tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                                (preds == cls_idx).cpu().numpy().astype(int))
                    tp_plus_fp = np.sum((preds == cls_idx).cpu().numpy())
                    tp_plus_fn = np.sum((labels.cpu().data == cls_idx).numpy())
                    # log_dict[cls_idx] = {'tp': 0, 'tp_fp': 0, 'tp_fn': 0}
                    log_dict[cls_idx]['tp'] += tp
                    log_dict[cls_idx]['tp_plus_fp'] += tp_plus_fp
                    log_dict[cls_idx]['tp_plus_fn'] += tp_plus_fn

                # 记录内容给tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' minibatch loss',
                                      float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' minibatch ACC',
                                      float(torch.sum(preds == labels.data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # 画图检测效果
                if index % check_minibatch == check_minibatch - 1:
                    model_time = time.time() - model_time

                    check_index = index // check_minibatch + 1

                    epoch_idx = epoch + 1
                    print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                          check_index, '     time used:', model_time)

                    print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

                    if enable_attention_check:
                        check_grad_CAM(model, check_dataloaders[phase], class_names, check_index, num_images=2, device=device,
                                       pic_name='GradCAM_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                       skip_batch=check_minibatch, draw_path=draw_path, writer=writer)
                    if enable_visualize_check:
                        visualize_check(model, check_dataloaders[phase], class_names, check_index, num_images=3,
                                        device=device,
                                        pic_name='Visual_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                        skip_batch=check_minibatch, draw_path=draw_path, writer=writer)

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            # 记录输出本轮情况
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            print('\nEpoch: {}  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))
            # 记录内容给tensorboard
            if writer is not None:
                # ...log the running loss
                writer.add_scalar(phase + ' loss',
                                  float(epoch_loss),
                                  epoch + 1)
                writer.add_scalar(phase + ' ACC',
                                  float(epoch_acc),
                                  epoch + 1)

            for cls_idx in range(len(class_names)):
                tp = log_dict[cls_idx]['tp']
                tp_plus_fp = log_dict[cls_idx]['tp_plus_fp']
                tp_plus_fn = log_dict[cls_idx]['tp_plus_fn']

                if tp_plus_fp == 0:
                    precision = 0
                else:
                    precision = float(tp) / tp_plus_fp * 100

                if tp_plus_fn == 0:
                    recall = 0
                else:
                    recall = float(tp) / tp_plus_fn * 100

                print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
                # 记录内容给tensorboard
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' precision',
                                      precision,
                                      epoch + 1)
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' recall',
                                      recall,
                                      epoch + 1)

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # 假设这里是train的时候，记得记录

            # deep copy the model，如果本epoch为止比之前都表现更好才刷新参数记录
            # 目的是在epoch很多的时候，表现开始下降了，那么拿中间的就行
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac):  # 需要定义"更好"
                best_epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_log_PR = log_dict
                best_model_wts = copy.deepcopy(model.state_dict())

            print('\n')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch idx: ', best_epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))
    print('Best epoch val PR:')  # best_log_PR
    for cls_idx in range(len(class_names)):
        tp = best_log_PR[cls_idx]['tp']
        tp_plus_fp = best_log_PR[cls_idx]['tp_plus_fp']
        tp_plus_fn = best_log_PR[cls_idx]['tp_plus_fn']

        if tp_plus_fp == 0:
            precision = 0
        else:
            precision = float(tp) / tp_plus_fp * 100

        if tp_plus_fn == 0:
            recall = 0
        else:
            recall = float(tp) / tp_plus_fn * 100

        print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))

    # 记录内容给tensorboard
    if writer is not None:
        writer.close()

    # load best model weights as final model training result 这也是一种避免过拟合的方法
    model.load_state_dict(best_model_wts)
    return model


def visualize_check(model, dataloader, class_names, check_index=1, num_images=9, device='cpu', skip_batch=10,
                    pic_name='test', draw_path='/home/ZTY/imaging_results', writer=None):  # 预测测试
    '''
    对num_images个图片进行检查，每行放3个图

    :param model:输入模型
    :param dataloader:输入数据dataloader
    :param class_names:分类的类别名字
    :param num_images:需要检验的原图数量,此数量需要小于batchsize
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

        dataloader = iter(dataloader)
        for i in range(check_index * skip_batch):
            inputs, classes = next(dataloader)

        inputs = inputs.to(device)
        labels = classes.to(device)

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

                '''
                myfig = plt.gcf()  # get current image
                myfig.savefig(picpath, dpi=1000)
                '''
                plt.savefig(picpath, dpi=1000)
                plt.show()

                model.train(mode=was_training)

                if writer is not None:  # 用这个方式读取保存的图片到tensorboard上面
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                return

        model.train(mode=was_training)


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(20)

    enable_notify = False
    enable_tensorboard = True
    enable_attention_check = True
    enable_visualize_check = False

    # 使用Agg模式，不在本地画图
    import matplotlib

    matplotlib.use('Agg')

    # 任务类别数量
    num_classes = 2
    model_type = '50'  # 用哪个ResNet 50 34 101？
    Transfer_learning = True

    # 边长
    edge_size = 384  # 1000

    # 训练设置
    batch_size = 4

    num_epochs = 50

    lr = 0.00001

    opt_name = 'Adam'  # 4用Adam   2用SGD

    model_idx = 'ResNet' + model_type + '_' + str(edge_size)+ '_401'
    model_ft = get_model(num_classes, edge_size, model_idx)

    # 配置
    draw_path = r'C:\Users\admin\Desktop\runs\PC_' + model_idx
    model_path = r'C:\Users\admin\Desktop\saved_models'
    save_model_path = model_path + '\PC_' + model_idx + '.pth'

    dataroot = r'C:\Users\admin\Desktop\ZTY_dataset'
    '''
    # 配置
    draw_path = '/home/ZTY/runs/PC_' + model_idx
    model_path = '/home/ZTY/saved_models'
    model_path = model_path + '/PC_' + model_idx + '.pth'
    
    dataroot = '/data/pancreatic-cancer-project/712_dataset'
    '''
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(draw_path)  # 每次开始的时候都先清空一次
    else:
        os.makedirs(draw_path)

    if enable_notify:
        import notify

        if enable_tensorboard:
            notify.add_text('进行CTE训练 编号：' + str(model_idx) + '  ' + str(num_classes) + '分类，优化器' + opt_name
                            + '_e' + str(num_epochs) + '.  上传到tensorboard')
        else:
            notify.add_text('进行CTE训练 编号：' + str(model_idx) + '  ' + str(num_classes) + '分类，优化器' + opt_name
                            + '_e' + str(num_epochs) + '.  过程没有上传到tensorboard')
        notify.add_text('边长 edge_size =' + str(edge_size))
        notify.add_text('batch_size =' + str(batch_size))
        notify.send_log()
    else:
        # 调取tensorboard服务器
        if enable_tensorboard:
            writer = SummaryWriter(draw_path)
        else:
            writer = None
    # writer = SummaryWriter(draw_path)
    # nohup tensorboard --logdir=/home/ZTY/runs --host=10.201.10.16 --port=7777 &
    # tensorboard --logdir=C:\Users\admin\Desktop\runs --host=192.168.1.139 --port=7777

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.CenterCrop(700),  # 旋转之后选取中间区域（避免黑边）
            transforms.Resize(edge_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
            # 色相饱和度对比度明度的相关的处理H S L，随即灰度化
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(700),
            transforms.Resize(edge_size),
            transforms.ToTensor()
        ]),
    }
    # 注意，这里的预处理参数是按照预训练的配置来设置的，这样迁移学习才能有更明显的效果
    # 最后加上 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    lung2d_datasets = {x: datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {'train': torch.utils.data.DataLoader(lung2d_datasets['train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=1),
                   'val': torch.utils.data.DataLoader(lung2d_datasets['val'], batch_size=batch_size, shuffle=False,
                                                      num_workers=1)
                   }

    # 需要根据任务修改自己的类别名字
    class_names = ['negative', 'positive'][0:num_classes]  # A G E B
    dataset_sizes = {x: len(lung2d_datasets[x]) for x in ['train', 'val']}  # 数据数量

    # Decide which device we want to run on
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只让程序看到物理卡号为card_no的卡（注意：no标号从0开始）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 这个是为了之后走双卡
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 只让程序看到物理卡号为card_no的卡（注意：no标号从0开始）
    # device = torch.device('cuda:0')  # 逻辑卡号cuda：0调用单卡

    '''
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)
    '''
    model_ft.to(device)

    summary(model_ft, input_size=(3, edge_size, edge_size))  # to device 之后安排, 输出模型结构

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00005, momentum=0.9)
    # Every step_size epochs, Decay LR by multipling a factor of gamma
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Adam
    # optimizer = optim.Adam(model_ft.parameters(), lr=0.0000001, weight_decay=0.01)
    # scheduler = None

    # SGD
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.00002, momentum=0.8)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    if opt_name == 'SGD':
        optimizer = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.8)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 15 0.1
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=0.01)
        scheduler = None

    # train，路径配置的16号服务器的路径，下一步需要给这个train加grad cam
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer, class_names, dataset_sizes,
                           num_epochs=num_epochs, check_minibatch=100, scheduler=scheduler, device=device,
                           draw_path=draw_path, enable_attention_check=enable_attention_check,
                           enable_visualize_check=enable_visualize_check, writer=writer)
    # 保存模型
    torch.save(model_ft.state_dict(), save_model_path)
