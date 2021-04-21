'''
这个虽然对上了coco的数据dataloader，但是我不确定是否正确的匹配格式在工作

'''

import os
import numpy as np
import torch
from PIL import Image

# os.environ['CUDA_VISIBLE_DEVICES']='1'
'''
输入数据PipeLine
pytorch 的数据加载到模型的操作顺序是这样的：

① 创建一个 Dataset 对象
② 创建一个 DataLoader 对象
③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练

Dataset
一个torch需要的dataset，主要有3个函数，__init__，__getitem__，__len__。
分别对应：初始化，按idx获得一个数据，获得总数量



先介绍一下DataLoader(object)的参数：

dataset(Dataset): 传入的数据集

batch_size(int, optional): 每个batch有多少个样本

shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序

sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False

batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，
那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）

num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）

collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数

pin_memory (bool, optional)： 
如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.

drop_last (bool, optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，
那么训练的时候后面的36个就被扔掉了…如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

timeout(numeric, optional): 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。
这个numeric应总是大于等于0。默认为0

worker_init_fn (callable, optional): 每个worker初始化函数 If not None, this will be called on each
worker subprocess with the worker id (an int in [0, num_workers - 1]) as
input, after seeding and before data loading. (default: None)
————————————————
原文链接：https://blog.csdn.net/g11d111/article/details/81504637
'''

# 构建Dataloader
from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import presets
import utils


def get_dataset(name, image_set, transform, data_path, num_classes=3):
    paths = {
        "coco": (data_path, get_coco, num_classes),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform_coco(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()



# 重写coco dataloader参考 https://www.cnblogs.com/zi-wang/p/9972102.html


'''
 two common situations where one might want to modify one of the available models in torchvision modelzoo. 
 The first is when we want to start from a pre-trained model, and just finetune the last layer. 
 The other is when we want to replace the backbone of the model with a different one (for faster predictions, etc.).
'''

# 1 - Finetuning from a pretrained model

'''
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (person) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
'''

# 2 - Modifying the model to add a different backbone

'''
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features

# FasterRCNN needs to know the number of output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will use to perform the region of interest cropping,
# as well as the size of the crop after rescaling.

# if your backbone returns a Tensor, featmap_names is expected to be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
'''

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


'''
# 测试一下dataloader

coco_path = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco'

dataset, _ = get_dataset('coco', "train", get_transform_coco(train=True), coco_path, num_classes=3)
dataset_test, _ = get_dataset('coco', "val", get_transform_coco(train=False), coco_path, num_classes=3)

train_sampler = torch.utils.data.RandomSampler(dataset)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)
train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, 8, drop_last=True)  # batch_size  8

data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=5,  # args.workers并行做数据采样的线程数
        collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2,
        sampler=test_sampler, num_workers=5,  # args.workers
        collate_fn=utils.collate_fn)

images, targets = next(iter(data_loader))

images = list(image for image in images)

targets = [{k: v for k, v in t.items()} for t in targets]

# 测试一下模型能不能用，forward是否正常
# get the model using our helper function
model = get_model_instance_segmentation(num_classes=3)

output = model(images, targets)  # Returns losses and detections
print(output,'\n')

# For inference
model.eval()
predictions = model(images) # Returns predictions

print(predictions)
'''


from engine import train_one_epoch, evaluate
import utils


def main():
    # train on the GPU or on the CPU, if a GPU is not available

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # 现在不知道什么情况，gpu就是用不了，engine里面的train one epoch会有问题

    # our dataset has 3 classes only
    num_classes = 3  # 背景 与 A 和 G ，  coco数据集内编码为0不可用，代表着背景，A为1，G为2

    # use our dataset and defined transformations, data saved at PATH
    coco_path = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco'

    dataset, _ = get_dataset('coco', "train", get_transform_coco(train=True), coco_path, num_classes=num_classes)
    dataset_test, _ = get_dataset('coco', "val", get_transform_coco(train=False), coco_path, num_classes=num_classes)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, 8, drop_last=True)  # batch_size  8

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=5,  # args.workers并行做数据采样的线程数
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2,
        sampler=test_sampler, num_workers=5,  # args.workers
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


main()
