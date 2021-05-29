import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

from coco_imaging import coco_a_result_check


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        if epoch - epoch // print_freq * print_freq == 0:
            pass

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


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
def evaluate(model, data_loader, device, epoch_num=None, check_num=10, save_pic_path='../ZTY/imaging_results'):
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

    for images, targets in metric_logger.log_every(data_loader, check_num, header):
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
        if idx - idx // check_num * check_num == 0:  # 每100次记录一次
            if epoch_num is not None:
                coco_a_result_check(images, targets, res, 'E' + str(epoch_num) + '_' + str(idx),
                                    save_pic_path=save_pic_path)
            else:
                coco_a_result_check(images, targets, res, save_pic_path=save_pic_path)

        '''
        for key in res:
            print(len(res[key]['boxes']))  # 一开始mask rcnn网络输出是100个框（detr_ori = object enquries），
            # 后续学好了之后框的数量会大大下降rcnn用非极大抑制，DETR用匈牙利算法。
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
