"""
版本：3月16日  00:20

暂时张天翊测试coco的可视化
"""

import os
import sys

# 将当前目录和父目录加入路径，使得文件可以调用本目录和父目录下的所有包和文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch


# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".
def check_name_dict(label_idx):
    # 为了计算速度，把dict放这里
    label_idx = str(label_idx)
    name_dict = {'1': 'tumor'}
    # name_dict = {'1': 'person'}
    # name_dict = {'1': 'ADC', '2': 'SCC'}
    # name_dict = {'1': 'ADC', '2': 'SCC', '3': 'LDC', '4': 'NOS'}

    try:
        return name_dict[label_idx]
    except:
        return 'Not required'


def coco_2dGT_check(annFile_path, picFile_path, imgId,
                    save_pic_path="/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/imging_results"):
    """
    对ground truth进行可视化检查
    :param save_pic_path: 新图路径 '/home/NSCLC-project/NSCLC_go/COCO_NSCLC/imging_results'
    :param annFile_path: json绝对路径
    :param picFile_path: image绝对路径
    :param imgId: 测试的图像编号
    :return:
    """
    if not os.path.exists(save_pic_path):
        os.makedirs(save_pic_path)

    coco = COCO(annFile_path)  # initialize COCO api for instance annotations
    imgInfo = coco.loadImgs(imgId)[0]
    # print(f'图像{imgId}的信息如下：\n{imgInfo}')

    imPath = os.path.join(picFile_path, imgInfo['file_name'])
    im = cv2.imread(imPath)
    im = np.asarray(im)

    # 获取该图像对应的一系列anns的Id
    annIds = coco.getAnnIds(imgIds=imgInfo['id'])
    # print(f'图像{imgInfo["id"]}包含{len(annIds)}个ann对象，分别是:\n{annIds}')
    anns = coco.loadAnns(annIds)

    # 对应的mask
    masks_of_a_slice = np.zeros_like(im)
    for ann in anns:
        mask = coco.annToMask(ann)  # 01mask
        mask = np.asarray(mask)
        mask = np.stack((mask,) * 3, axis=-1)  # 转化为3通道,CV2的格式为h w c=3

        masks_of_a_slice += mask  # 理论上不会重叠1区域，但是如果是预测输出则要改一下再用?

    heatmap = np.uint8(255 * masks_of_a_slice)  # 对数值进行色域修改赋值，色域在 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将数值变成颜色

    superimopsed_img = heatmap * 0.4 + im  # 涂上去的热力图颜色深度需要调整，之后进行叠加

    # 对应的boxs测试
    for ann in anns:
        # 加文本参考：https://www.cnblogs.com/shuangcao/p/11344436.html
        # coco框参考： https://blog.csdn.net/chao_shine/article/details/107581345
        cv2.putText(superimopsed_img, coco.loadCats(ann['category_id'])[0]['name'],
                    (ann['bbox'][0], ann['bbox'][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        x1 = int(ann['bbox'][0])
        y1 = int(ann['bbox'][1] + ann['bbox'][3])
        x2 = int(ann['bbox'][0] + ann['bbox'][2])
        y2 = int(ann['bbox'][1])

        cv2.rectangle(superimopsed_img,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      2)

    img_path = os.path.join(save_pic_path, str(imgId) + '.png')
    cv2.imwrite(img_path, superimopsed_img)  # 保存图像数组为灰度图(.png)
    return img_path


def coco_a_result_check(images, targets, res, idx='None',
                        save_pic_path="/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/imaging_results"):
    if not os.path.exists(save_pic_path):
        os.makedirs(save_pic_path)

    check_num = len(images)

    for i in range(check_num):
        a_image = images[i]
        a_target = targets[i]
        image_id = a_target["image_id"].item()

        paint_a_coco_graph(a_image, a_target,
                           save_pic_path + '/' + idx + 'True_on_' + str(image_id) + '_' + str(i) + '.png')

        for key in res:
            if int(key) == image_id:
                b_target = res[key]
                paint_a_coco_graph(a_image, b_target,
                                   save_pic_path + '/' + idx + 'Pred_on_' + str(image_id) + '_' + str(i) + '.png')


def paint_a_coco_graph(a_image, a_target, img_path, device='cpu'):

    a_image = a_image.to(device).numpy()
    a_image = a_image * 255  # 模型拿到的数据，做了正太化，这里需要重新转回255可视化模式

    a_image = a_image.transpose(1, 2, 0).astype(np.uint8).copy()  # 转为cv2格式

    target_items_num = len(a_target['boxes'])  # 识别到的目标数

    # total label
    bboxs = a_target['boxes']
    labels = a_target['labels']

    # 如果有mask
    mask_status = False
    for key in a_target:
        if key == 'masks':
            mask_status = True
            # total mask
            masks_of_a_slice = np.zeros_like(a_image)
            for target_item_idx in range(target_items_num):
                mask = a_target['masks'][target_item_idx]  # 2D binary mask
                mask = mask.to(device).detach().numpy()

                mask = np.stack((mask,) * 3, axis=-1)  # 转化为3通道,CV2的格式为h w c=3
                # mask = mask > 0

                masks_of_a_slice = masks_of_a_slice + mask

            masks_of_a_slice = masks_of_a_slice > 0
            masks_of_a_slice = masks_of_a_slice.squeeze(axis=None)

            heatmap = np.uint8(255 * masks_of_a_slice)  # 对数值进行色域修改赋值，色域在 0-255

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将数值变成颜色

            superimopsed_img = heatmap * 0.2 + a_image  # 涂上去的热力图颜色深度需要调整，之后进行叠加

    if mask_status:
        superimopsed_img = superimopsed_img
    else:
        superimopsed_img = a_image

    for target_item_idx in range(target_items_num):
        # 加文本参考：https://www.cnblogs.com/shuangcao/p/11344436.html
        # coco框参考： https://blog.csdn.net/chao_shine/article/details/107581345
        cv2.putText(superimopsed_img, check_name_dict(label_idx=int(labels[target_item_idx])),
                    (int(bboxs[target_item_idx][0]), int(bboxs[target_item_idx][1])),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=1)

        x1 = int(bboxs[target_item_idx][0])  # xmin
        y1 = int(bboxs[target_item_idx][3])  # ymax
        x2 = int(bboxs[target_item_idx][2])  # xmax
        y2 = int(bboxs[target_item_idx][1])  # ymin

        cv2.rectangle(superimopsed_img,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      1)
    cv2.imwrite(img_path, superimopsed_img)  # 保存图像数组为灰度图(.png)


def main():
    cocoRoot = "/media/jefftian/44f36ce2-18b3-4775-952e-6152eedda284/ZTY/data/coco"
    dataType = "val2017"

    annFile_path = os.path.join(cocoRoot, f'annotations/instances_{dataType}.json')
    picFile_path = os.path.join(cocoRoot, dataType)  # coco格式有images则需要这个(cocoRoot, 'images', dataType)

    coco = COCO(annFile_path)

    # 确定id！！！！
    imgId = 14871  # coco.catToImgs[2][101]

    # 查询数据信息
    imgInfo = coco.loadImgs(imgId)[0]
    print(imgInfo)

    # 画原图，此时从coco拿到的是255的图，能够直接画，但是如果进入dataloader拿到的就是normalized的图片，需要乘255
    img_path = coco_2dGT_check(annFile_path, picFile_path, imgId)


# main()
