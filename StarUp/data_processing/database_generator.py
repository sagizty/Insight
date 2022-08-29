"""
Organize the data to ensure that all data is in jpg format  ver： Jan 9th 15：30 official release

数据集地址：https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download
"""

import os
import re
import csv
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def del_file(filepath):
    """
    Delete all files and folders in one directory
    :param filepath: file path
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


def trans_csv_folder_to_imagefoder(target_path=r'C:\Users\admin\Desktop\MRAS_SEED_dataset',
                                   original_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_org_image',
                                   csv_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_label.csv'):
    """
    Original data format: a folder with image inside + a csv file with header which has the name and category of every image.
    Process original dataset and get data packet in image folder format

    :param target_path: the path of target image folder
    :param original_path: The folder with images
    :param csv_path: A csv file with header and the name and category of each image
    """
    idx = -1
    with open(csv_path, "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

        make_and_clear_path(target_path)  # Clear target_path

        for row in tqdm(rows):
            idx += 1

            if idx == 0:  # Skip the first header
                continue

            item_path = os.path.join(original_path, row[0])
            # todo 这个【0【1】要修改
            if os.path.exists(os.path.join(target_path, row[2])):
                shutil.copy(item_path, os.path.join(target_path, row[2]))
            else:
                os.makedirs(os.path.join(target_path, row[2]))
                shutil.copy(item_path, os.path.join(target_path, row[2]))

        print('total num:', idx)


if __name__ == '__main__':
    trans_csv_folder_to_imagefoder()
