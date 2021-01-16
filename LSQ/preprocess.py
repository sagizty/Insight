'''
预处理：
读取图片数据和对应的标签，并转化为numpy保存在npy_path目录下
（比较慢，只需运行一次）
'''

import numpy as np
import os
import glob
from skimage import transform, io
import shutil

def preprocess(img_path, npy_path, img_rows, img_cols):

    '''
    读取图片，并转化为numpy保存到指定位置
    :param img_path: 图片位置
    :param npy_path: 生成的numpy保存位置
    :param img_rows: 图片w
    :param img_cols: 图片h
    :return:
    '''

    # 读取所有图片分类目录
    print('start preprocessing...')
    data_list = [os.path.join(img_path, x) for x in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, x))]
    imgs = []  # 定义一个imgs空列表，存放遍历读取的图片
    labels = []  # 定义一个labels空列表，存放图片标签
    for idx, folder in enumerate(data_list):  # 遍历每个文件夹中的图片，idx表示
        for image in glob.glob(folder + '/*.jpg'):  # *:匹配0个或多个字符
            img = io.imread(image)
            img = transform.resize(img, (img_rows, img_cols))  # 将所有图片的尺寸统一为:100*100(宽度*高度)
            imgs.append(img)  # 遍历后更改尺寸后的图片添加到imgs列表中
            labels.append(idx)  # 遍历后更改尺寸后的图片标签添加到labels列表中

    # 打乱并划分数据集
    x_train, y_train, x_test, y_test = shuffle_data(np.asarray(imgs, np.float32), np.asarray(labels, np.int32))
    x_train /= 255
    x_test /= 255
    # y_train = tf.keras.utils.to_categorical(y_train, num_classes)[:600]
    # y_test = tf.keras.utils.to_categorical(y_test, num_classes)[:100]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # 保存结果
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
    os.mkdir(npy_path)
    np.save(os.path.join(npy_path, 'x_train.npy'), x_train)
    np.save(os.path.join(npy_path, 'y_train.npy'), y_train)
    np.save(os.path.join(npy_path, 'x_test.npy'), x_test)
    np.save(os.path.join(npy_path, 'y_test.npy'), y_test)
    return


def shuffle_data(data, label):
    '''
    随机打乱数据集
    :param data:
    :param label:
    :return:
    '''

    # 打乱顺序
    data_size = data.shape[0]  # 数据集个数
    arr = np.arange(data_size)  # 生成0到datasize个数
    np.random.shuffle(arr)  # 随机打乱arr数组
    data = data[arr]  # 将data以arr索引重新组合
    label = label[arr]  # 将label以arr索引重新组合

    #    # 打乱数据顺序的另一种方法，当然还有其他的方法
    #    index = [i for i in range(len(data))]
    #    random.shuffle(index)
    #    data = data[index]
    #    label = label[index]

    # 将所有数据分为训练集和验证集
    ratio = 0.8  # 训练集比例
    num = np.int(len(data) * ratio)
    x_train = data[:num]
    y_train = label[:num]
    x_val = data[num:]
    y_val = label[num:]

    return x_train, y_train, x_val, y_val



# 数据：http://download.tensorflow.org/example_images/flower_photos.tgz
# 花总共有五类，分别放在5个文件夹下。
img_path = 'flower_photos'     # 图片路径
npy_path = 'npy_data'       # numpy数组存储路径
img_rows, img_cols = 100, 100     # 图片resize的宽度和高度, 准备将所有的图片resize成100*100

# 进行预处理
preprocess(img_path, npy_path, img_rows, img_cols)