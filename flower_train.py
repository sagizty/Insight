# -*- coding: utf-8 -*-
"""
参考资料：
1。Tensorflow框架搭建卷积神经网络进行五种花的分类
https://blog.csdn.net/AugustMe/article/details/94166164
"""

import numpy as np
import os
import glob
from skimage import transform, io
from keras import layers
from keras import models
from keras import optimizers, losses
import matplotlib.pyplot as plt
from keras.utils import to_categorical


# 定义读取图片的函数：read_img()
def processing_data(all_data_path, l=224, w=224, font='/'):
    """
    读取全部数据，转化为一个数据集npy与一个标签集npy
    :param all_data_path:
    :param l:
    :param w:
    :param font:
    :return:
    """
    # 所有图片分类目录
    data_list = [all_data_path + font + x for x in os.listdir(all_data_path) if os.path.isdir(all_data_path + font + x)]
    imgs = []  # 定义一个imgs空列表，存放遍历读取的图片
    labels = []  # 定义一个labels空列表，存放图片标签

    for idx, folder in enumerate(data_list):  # 遍历每个文件夹中的图片，idx表示

        for im in glob.glob(folder + '/*.jpg'):  # *:匹配0个或多个字符
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (l, w))  # 将所有图片的尺寸统一为:w*h(宽度*高度)

            with open('datasets_name.txt', 'a') as f:
                f.write(folder + im + '_' + str(idx) + '\n')
            imgs.append(img)  # 遍历后更改尺寸后的图片添加到imgs列表中
            labels.append(idx)  # 遍历后更改尺寸后的图片标签添加到labels列表中

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)  # np.float32是类型 后面两个变量是没有进行np.asarray


# 定义随机打乱数据集的函数：shuffle_data()
def shuffle_data(data, label, ratio=0.8):  # 训练集比例 ratio = 0.8
    # 打乱顺序
    data_size = data.shape[0]  # 数据集个数
    arr = np.arange(data_size)  # 生成0到datasize个数
    np.random.shuffle(arr)  # 随机打乱arr数组
    data = data[arr]  # 将data以arr索引重新组合
    label = label[arr]  # 将label以arr索引重新组合

    num = np.int(len(data) * ratio)
    # x是数据，y是标签
    x_train = data[:num]  # 训练集
    y_train = label[:num]
    x_val = data[num:]  # 测试集
    y_val = label[num:]

    return x_train, x_val, y_train, y_val


def uploading(data, label, feature_dim_1=224, feature_dim_2=224, channel=3):
    """
    将数据转换为适合cnn的输入格式
    :param data: 数据npy
    :param label: 标签npy
    :param feature_dim_1: 维度1
    :param feature_dim_2: 维度2
    :param channel: 通道（维度3）
    :return:
    """

    x_train, x_test, y_train, y_test = shuffle_data(data, label)

    x_train = x_train.reshape(x_train.shape[0], feature_dim_1, feature_dim_2, channel)  # 数据数量，维度，宽度，核通道数（颜色）
    x_test = x_test.reshape(x_test.shape[0], feature_dim_1, feature_dim_2, channel)

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    return x_train, x_test, y_train_hot, y_test_hot


def draw_training_summary(history, picpath='./training result.jpg'):
    """
    画训练过程中的记录
    :param history:
    :param picpath:
    :return:
    """
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(picpath)
    plt.show()


# 模型
def vgg_model(feature_dim_1=224, feature_dim_2=224, channel=3, target=5):
    """
    模型
    :param feature_dim_1:
    :param feature_dim_2:
    :param channel:
    :param target: 分类数，与结构有关
    :return:
    """
    model = models.Sequential()
    model.build(input_shape=(None, feature_dim_1, feature_dim_2, channel))

    model.add(layers.SeparableConv2D(64, channel, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.SeparableConv2D(128, channel, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.SeparableConv2D(256, channel, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.SeparableConv2D(512, channel, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.SeparableConv2D(512, channel, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        layers.SeparableConv2D(512, channel, padding='SAME', activation='relu'))  # layername = r'separable_conv2d_6'
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(target, activation='softmax'))

    # model.summary()
    # 这里可以知道每层layer的名字！

    return model


def begin(project_data_path, all_data_path, model_h5name, process_data=True, epochs=15, l=224, w=224, c=3, target=5,
          font='/'):
    """
    总函数
    :param project_data_path: 项目文件路径
    :param all_data_path: 项目数据路径
    :param model_h5name: 存放的模型名字

    :param process_data: 是否读取数据，若为否，则读取之前处理过整理好的数据集npy

    :param epochs:
    :param l:
    :param w:
    :param c:
    :param target: 分类个数
    :param font:
    :return:
    """
    # 处理数据
    if process_data:
        data, label = processing_data(all_data_path, l, w, font=font)
        np.save(project_data_path + font + "data.npy", data)
        np.save(project_data_path + font + "label.npy", label)
    else:
        data = np.load(project_data_path + font + "data.npy")
        label = np.load(project_data_path + font + "label.npy")

    # 调模型，编译模型
    model = vgg_model(l, w, c, target)
    print('model is ok now')
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])
    # model = k_folds_verify(model, data, label, k=10)

    # 装载数据
    x_train, x_test, y_train_hot, y_test_hot = uploading(data, label, l, w, c)

    # 训练
    # plot(model, to_file='model1.png', show_shapes=True) #keras模型可视化
    history = model.fit(x_train, y_train_hot, batch_size=100, epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test_hot))

    # 保存训练结果+画图呈现训练效果
    model_path = project_data_path + font + model_h5name
    model.save(model_path)

    picpath = project_data_path + font + 'training result.jpg'
    draw_training_summary(history, picpath)


# 定义超参数
l = 224
w = 224
c = 3  # 颜色
target = 5  # 花的种类数量

project_data_path = '/Users/zhangtianyi/Study/cam_flowers'
all_data_path = '/Users/zhangtianyi/Study/cam_flowers/train_data'  # 所有图片的总路径(目录)
model_h5name = 'five_flowers_categorical_vgg16.h5'
plot_pic_path = 'training result.jpg'

begin(project_data_path, all_data_path, model_h5name, True, 15, l, w, c, font='/')
