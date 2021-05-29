import os
import numpy as np
import torch
from PIL import Image
import shutil


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


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)
    del_file(file_pack_path)
    # print("Empty path create at: ",file_pack_path)


# os.environ['CUDA_VISIBLE_DEVICES']='0'
'''
输入数据PipeLine
pytorch 的数据加载到模型的操作顺序是这样的：

① 创建一个 Dataset 对象
② 创建一个 DataLoader 对象
③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练

Dataset
一个torch需要的dataset，主要有3个函数，__init__，__getitem__，__len__。
分别对应：初始化，按idx获得一个数据，获得总数量


'''


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        # 这里传个来自pytorch的transform函数实现数据变换
        self.transforms = transforms

        # load all image files path, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):  # 按idx取一个数据, 从0开始
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance, with 0 being background

        # convert the PIL Image into a numpy array
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd, instances with iscrowd=True will be ignored during evaluation.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # create return anno，返回的是一个图片的一系列标注结果
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):  # 总长度
        return len(self.imgs)


'''
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


# 需要把这几个文件找到放到同一个文件夹
# references/detection/engine.py, references/detection/utils.py and references/detection/transforms.py

import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


import utils, engine



'''
PATH='/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/PennFudanPed'
# 构建dataloader
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

dataset = PennFudanDataset(PATH, get_transform(train=True))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                          collate_fn=utils.collate_fn)



# 测试一下模型能不能用，forward是否正常
# For Training
images, targets = next(iter(data_loader))

images = list(image for image in images)

targets = [{k: v for k, v in t.items()} for t in targets]

print(targets,'\n')
print(len(targets),'\n')  # 对应batch
print('每个target 内项目条目',len(targets[0]),'\n')  # 对应target 内容
print('对应target 1 的目标数',len(targets[0]['masks']),'\n')  # 对应target 内容
print('对应target 2 的目标数',len(targets[1]['masks']),'\n')  # 对应target 内容

# 测试faster Rcnn
print("\n\n测试faster Rcnn")
output = model(images, targets)  # Returns losses and detections
print(output,'\n')  # 一个字典，包含4种loss的记录
print('每个output 内条目',len(output),'\n')  # 4


# For inference
model.eval()
predictions = model(images) # 对应batch ，Returns predictions

print(predictions,'\n')
print('predictions 个数对应batch',len(predictions),'\n')
print('每个predictions 内条目',len(predictions[0]),'\n')
print('预测目标个数',len(predictions[0]['scores']),'\n')  # 不一定，propose出来很多，选择掉一部分，还会剩余一些



print("现在把模型换为全景分割的头\n\n")

# 现在把模型换为全景分割的头
model = get_model_instance_segmentation(num_classes=2) # 1背景+1人
output = model(images, targets)  # Returns losses and detections
print(output,'\n')  # 一个字典，包含5种loss的记录
print(len(output),'\n')  # 5


# For inference
model.eval()
predictions = model(images) # 对应batch ，Returns predictions

print(predictions,'\n')
print('predictions 个数对应batch',len(predictions),'\n')
print('每个predictions 内条目',len(predictions[0]),'\n')
print('预测目标个数', len(predictions[0]['scores']),'\n')
print('分割目标个数', len(predictions[0]['masks']),'\n')
'''


from engine import train_one_epoch, evaluate
import utils


def main():
    PATH = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/PennFudanPed'
    # PATH = '/home/ZTY/data/PennFudanPed'

    save_pic_path = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/Fudan_imaging'

    make_and_clear_path(save_pic_path)

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # 现在不知道什么情况，16号gpu就是用不了，engine里面的train one epoch会有问题, 如下：
    # RuntimeError: radix_sort: failed on 1st step: cudaErrorInvalidDevice: invalid device ordinal

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations, data saved at PATH
    dataset = PennFudanDataset(PATH, get_transform(train=True))
    dataset_test = PennFudanDataset(PATH, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    '''
    # 用肿瘤的拿过来
    model.load_state_dict(torch.load('/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/'
                                     + 'saved_models/maskrcnn_resnet50_2_ins_seg_S2.pth'))
    '''

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    # 先进行pretrain效果记录
    evaluate(model, data_loader_test, check_num=20, device=device, save_pic_path=save_pic_path)

    print("done")

    for epoch in range(1, num_epochs+1):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, check_num=20, device=device, epoch_num = epoch, save_pic_path=save_pic_path)

    print("That's it!")

main()