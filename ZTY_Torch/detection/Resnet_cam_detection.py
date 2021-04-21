"""
使用Resnet50的分类CAM响应来实现定位，测试bbox效果

采用
"""

import sys
import os

# 将当前目录和父目录加入路径，使得文件可以调用本目录和父目录下的所有包和文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
import math

# from __future__ import print_function, division

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy

import torchvision.models.detection.mask_rcnn

# 来自torch官方的例子里面的辅助文件
import transforms as T
import utils
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from coco_imaging import coco_a_result_check


# 采用coco格式的数据
class coco_background_Dataset(object):
    def __init__(self, coco_root, datasettype, transforms=None, num_classes=2):
        """

        :param coco_root:
        :param model: train or val
        :param transforms:
        """
        # 这里传个来自pytorch的transform函数实现数据变换
        self.transforms = transforms

        self.annpath = os.path.join(coco_root, "annotations", 'instances_' + datasettype + '2017.json')
        self.image_path = os.path.join(coco_root, datasettype + "2017")

        self.coco = COCO(self.annpath)
        self.num_classes = num_classes

        self.image_ids = self.coco.getImgIds()

    def __getitem__(self, idx):  # 按tumor_slices_id取一个数据

        tumor_slices_id = self.image_ids[idx]

        imgInfo = self.coco.loadImgs(tumor_slices_id)[0]  # 【0】用于取出元素
        # print(f'图像{imgId}的信息如下：\n{imgInfo}')

        imPath = os.path.join(self.image_path, imgInfo['file_name'])

        # load image
        img = Image.open(imPath).convert("RGB")

        # 获取该图像对应的一系列anns的Id
        annIds = self.coco.getAnnIds(imgIds=imgInfo['id'])
        # print(f'图像{imgInfo["id"]}包含{len(annIds)}个ann对象，分别是:\n{annIds}')
        anns = self.coco.loadAnns(annIds)

        num_objs = len(anns)
        masks = []
        boxes = []
        labels = []

        for ann in anns:
            mask = self.coco.annToMask(ann)  # 01mask
            mask = np.asarray(mask)  # 每个ann id对应一个目标

            # coco mask是polygon格式编码的，不是01mask
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])  # 与coco格式不同！！！！！！！！！！！！
            # COCO_bbox = [xmin, ymin, width, height]   左上角横坐标、左上角纵坐标、宽度、高度
            masks.append(mask)
            label = int(self.coco.loadCats(ann['category_id'])[0]['id'])
            labels.append(label)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class：Tumor， so set the labels to 1
        if self.num_classes == 2:
            labels = torch.ones((num_objs,), dtype=torch.int64)
        else:
            labels = torch.as_tensor(labels, dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([tumor_slices_id])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 格式和coco不同！！

        # suppose all instances are not crowd, instances with iscrowd=True will be ignored during evaluation.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # create return anno
        target = {}

        target["boxes"] = boxes

        target["labels"] = labels
        target["masks"] = masks

        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 这个返回意味着，一个image 对应1个target，但是每个target内部的长度不确定（但是一致）

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):  # 总长度
        return len(self.image_ids)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


'''
# 测试数据加载
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

PATH = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco'
model_path = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/saved_models'

# comparing woth our dataset model's classification classes has one more classe - background as 0
num_classes = 3

# use our dataset and defined transformations, data saved at PATH
dataset = coco_background_Dataset(PATH, 'train', get_transform(train=True), num_classes=num_classes)
dataset_test = coco_background_Dataset(PATH, 'val', get_transform(train=False), num_classes=num_classes)

# 构建Dataloader
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)  # 走的是cpu，不要太高了
    
    
# 测试数据
images, targets = next(iter(data_loader))
images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
print("ok")
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


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=125, check_num=20,
                device=None, coco_1=-1, draw_path = '/home/ZTY/imaging_results'):
    # dataloaders是train和test dataloader的dict={'train':data_loader, 'val':data_loader_test}

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
            model_time = 0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            dataset_batchs = len(dataloaders[phase])
            print(phase + ' dataset_batchs ', dataset_batchs)

            # 记录表现
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for images, targets in dataloaders[phase]:  # 不同任务段用不同dataloader的数据

                inputs = torch.tensor([item.cpu().detach().numpy() for item in images]).to(device)

                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                labels = []
                for target in targets:
                    labels.append(target['labels'][0] + coco_1)  # + -1 是应为没有背景这个分类任务
                labels = torch.tensor(labels).to(device)

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

                if index - index // check_num * check_num == 0:
                    model_time = time.time() - model_time

                    epoch_idx = epoch + 1
                    print('Epoch:', epoch_idx, '    index of 100 minibatch:', index, '     time used:', model_time)
                    print('loss:', float(loss))
                    visualize_model(model, dataloaders, num_images=9, device=device, coco_1=coco_1,
                                    pic_name='E_' + str(epoch_idx) + '_I_' + str(index), draw_path =draw_path)
                    model_time = time.time()

                index += 1

                # 统计表现总和
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # 记录输出本轮情况
            epoch_loss = running_loss / dataset_batchs
            epoch_acc = running_corrects.double() / dataset_batchs
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


def grand_cam():
    pass


def coco_res_by_cam():
    pass


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, epoch_num=None, check_num=200):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    idx = 0

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs_set = model(images)  # 输出：对应a validate batch里面的每一个输出组成的list

        outputs_list = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_set]  # 对于minibatch里面的每个output

        # outputs_list包含一个个 t 是 {'boxes','labels','scores','masks'}，每个的值都是一个tensor

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs_list)}
        # 构建一个dict，每个键为target["image_id"].item() 即imageid
        # 值为对应数据在模型预测的时候的输出t， 是 {'boxes','labels','scores','masks'}字典，
        # 其内每个的值都是一个tensor，长度=预测目标数

        idx += 1
        if idx - idx // check_num * check_num == 0:  # 每check_num次记录一次
            if epoch_num is not None:
                coco_a_result_check(images, targets, res, 'E' + str(epoch_num) + '_' + str(idx))
            else:
                coco_a_result_check(images, targets, res)

        '''
        for key in res:
            print(len(res[key]['boxes']))  # 一开始mask rcnn网络输出是100个框（detr 200），后续学好了之后框的数量会大大下降。
        '''

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def visualize_model(model, dataloaders, num_images=9, device='cpu', coco_1=-1, pic_name='test',
                    draw_path='/home/ZTY/imaging_results'):
    # 按照dataloader一组组来跑，全部预测测试
    name_dict = {1 + coco_1: 'ADC', 2 + coco_1: 'SCC'}

    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloaders['val']):
            inputs = torch.tensor([item.cpu().detach().numpy() for item in images]).to(device)

            # print(inputs.shape) batch 3 512 512

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            labels = []
            for target in targets:
                labels.append(target['labels'][0] + coco_1)  # -1 是应为没有背景这个分类任务
            # labels = torch.tensor(labels).to(device)  gpu 上

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                plt.subplot(num_images // 3, 3, images_so_far)
                plt.axis('off')
                plt.title('Pred: {} True: {}'.format(name_dict[int(preds[j])], name_dict[int(labels[j])]),
                          fontdict={'weight': 'normal', 'size': 10})

                a_image = inputs.to('cpu').numpy()[j]

                a_image = a_image * 255  # 模型拿到的数据，做了正太化，这里需要重新转回255可视化模式
                a_image = a_image.transpose(1, 2, 0).astype(np.uint8).copy()  # 转为cv2格式

                plt.imshow(a_image)
                plt.pause(0.001)

                # a_image = inputs.cpu().data[j]
                # imshow(a_image)

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
                    return

        model.train(mode=was_training)


def main():
    import notify
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    PATH = '/home/NSCLC-project/Datasets/COCO_NSCLC_cls2_Ori_spacing/coco'
    # PATH = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco'
    model_path = '/home/ZTY/saved_models'
    # model_path = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/saved_models'
    draw_path = '/home/ZTY/imaging_results'
    # draw_path='/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/imaging_results'

    # comparing woth our dataset model's classification classes has one more classe - background as 0
    num_classes = 3

    # use our dataset and defined transformations, data saved at PATH
    dataset = coco_background_Dataset(PATH, 'train', get_transform(train=True), num_classes=num_classes)
    dataset_test = coco_background_Dataset(PATH, 'val', get_transform(train=False), num_classes=num_classes)

    # 构建Dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=25, shuffle=True, num_workers=5,
        collate_fn=utils.collate_fn)  # baiyan 设置18,4   16号25，5

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)  # 走的是cpu，不要太高了

    dataloaders = {'train': data_loader, 'val': data_loader_test}

    #  完全迁移学习：整个搞过来，换层，然后训练全部参数
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes - 1)  # -1 是应为没有背景这个分类任务，所以模型是2分类，此时coco num_classes = 3
    model = model.to(device)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 对应的，凡是采用coco数据建立的分类问题，需要在训练中对数据标签进行 减1

    #model.load_state_dict(torch.load(model_path + 'Resnet50_cam_detction.pth'))

    train_model(model, dataloaders, criterion, optimizer, scheduler_lr, num_epochs=200, check_num=200,
                device=device, coco_1=-1, draw_path=draw_path)

    # visualize_model(model, dataloaders, num_images=6, device=device, coco_1=-1)

    # 保存模型
    torch.save(model.state_dict(), model_path + 'Resnet50_cam_detction_200.pth')

    notify.send_log()


main()
