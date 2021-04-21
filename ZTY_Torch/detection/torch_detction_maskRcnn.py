'''
版本：4月 10日 mask Rcnn 迁移学习后训练

这个是采用faster/mask rcnn构建的检测

使用数据的规则格式和coco不同，因此需要转换！！！！！！！！！！！

主要是id编码coco是从1开始，以及，bbox的规则不同
匹配的是，双方都不采用0作为一个类的id，代表着背景

'''

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


# os.environ['CUDA_VISIBLE_DEVICES']='0'

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
            labels = torch.ones((num_objs,), dtype=torch.int64)  # 每个目标有几个ann就对应有几个1，eg：labels=【1，1，1】
        else:
            # 若是进行多类的识别，按类id就行，除了0类不能取！！！！！！！！！！
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


from engine import train_one_epoch, evaluate
import utils

'''
# 测试数据集是否能用


PATH='/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco'
# 测试构建dataloader
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

dataset = coco_background_Dataset(PATH,'train', get_transform(train=True))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                          collate_fn=utils.collate_fn)




# For Training
images, targets = next(iter(data_loader))

images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]


# 测试一下模型能不能用，forward是否正常
output = model(images, targets)  # Returns losses and detections
print(output)


# For inference
model.eval()
predictions = model(images)  # Returns predictions

print(predictions)




# 可视化效果测试
# 测试一下模型能不能用，forward是否正常
from coco_imaging import coco_a_result_check
cpu_device = torch.device("cpu")
# For Training
images, targets = next(iter(data_loader))

images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]


# For inference
model.eval()
outputs_set = model(images)  # 输出：对应a validate batch里面的每一个输出组成的list

outputs_list = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_set]
res = {target["image_id"].item(): output for target, output in zip(targets, outputs_list)}

coco_a_result_check(images, targets, res)

print('done')
'''


def main():
    import notify
    # train on the GPU or on the CPU, if a GPU is not available

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    PATH = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco'
    model_path = '/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/saved_models'

    # 现在不知道什么情况，16号的gpu就是用不了，engine里面的train one epoch会有问题

    # our dataset has two classes only - background and tumor
    num_classes = 2

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

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    try:
        model.load_state_dict(torch.load(model_path + '/maskrcnn_resnet50_' + str(num_classes) + '_ins_seg_S2.pth'))
    except:
        stage=1
    else:
        stage=3

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    # 最开始先看一下迁移过来不训练的时候之前有多菜
    evaluate(model, data_loader_test, device=device)
    print("Start training!")

    for epoch in range(1, num_epochs+1):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device, epoch_num=epoch)

    # 保存模型
    torch.save(model.state_dict(), model_path + '/maskrcnn_resnet50_' + str(num_classes) + '_ins_seg_S' + str(stage) +
    '.pth')

    print("That's it!")
    notify.send_log()


main()
