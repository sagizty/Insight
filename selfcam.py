"""
版本：12 29 02:39
这个框架是进行cam自监督
"""

import notify
import numpy as np
import os
import glob
# from skimage import transform, io
from keras import layers
from keras import models
from keras import optimizers, losses
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
import random

'''
# local host
# 项目数据位置
project_data_path = '/Users/zhangtianyi/Study/cam_flowers'

#big server 
project_data_path = '/home/FLOWER-project/Insight'

# little server host
# 项目数据位置
project_data_path = "/home/zty/Insight"
'''

font = '/'

# local host
# 项目数据位置
project_data_path = '/Users/zhangtianyi/Study/cam_flowers'

epochs = 3
width = 224
length = 224
std = 1
num_classes = 5


def split_data_multi_channel(data, label, percent=0.2):
    # data=[np.expand_dims(i,axis=-1) for i in data]  # 三维模型维
    # data = [i.transpose(1, 2, 0) for i in data] #二维模型
    data = np.asarray(data)
    l = int(data.shape[0] * 0.2)
    t = [i for i in range(data.shape[0])]
    temp = [j for j in set([random.randint(0, data.shape[0] - 1) for i in range(l)])]
    train_x = []
    label_x = []
    train_y = []
    label_y = []

    for i in temp:
        train_y.append(data[i])
        label_y.append(label[i])
        t.remove(i)

    for i in t:
        train_x.append(data[i])
        label_x.append(label[i])

    return np.asarray(train_x), np.asarray(label_x), np.asarray(train_y), np.asarray(label_y)


class VGG2D_Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.Sequential()

        for i in [64, 128, 256, 512, 512]:
            self.conv.add(layers.SeparableConv2D(i, 3, padding='same', activation='relu'))
            self.conv.add(layers.MaxPooling2D(pool_size=(2, 2)))
            self.conv.add(layers.BatchNormalization())

        self.linear = tf.keras.Sequential()
        self.linear.add(layers.Flatten())
        self.linear.add(layers.Dropout(0.5))
        self.linear.add(layers.Dense(32, activation='relu'))
        self.linear.add(layers.Dense(5))

    def call(self, x):
        conv = self.conv(x)
        pred = layers.Softmax()(self.linear(conv))

        return conv, pred


def core_deal(data, cam):
    cam = 1 + cam / (1 + sum(cam))  # 这里是把所有元素加和然后去做除法，为了防止sum（cam）等于0，选择加一个1，这样对整体不影响，其实可以用softmax，但是我特么不是懒得查么
    a = np.expand_dims(data[:, :, 0] * cam, axis=2)
    b = np.expand_dims(data[:, :, 0] * cam, axis=2)
    c = np.expand_dims(data[:, :, 0] * cam, axis=2)
    return np.concatenate((a, b, c), axis=2)


def to_loader(data, label, batch_size=64):
    loader = tf.data.Dataset.from_tensor_slices((data, label))
    loader = loader.shuffle(buffer_size=1000)
    loader = loader.batch(batch_size=batch_size)

    return loader
    # 读取数据


def cal_avg(l):
    a = 0
    c = 0
    for i in l:
        c += 1
        a += 1
    return a / c


def train(trainloader, testloader, model, epochs=epochs, std=std, num_classes=num_classes):
    datas = []
    labels = []
    cam_list = []

    for idx, (data, label) in enumerate(iter(trainloader)):
        for m, n in zip(data, label):
            datas.append(m)
            labels.append(n)

    optimizer = tf.keras.optimizers.Adam(0.0005, clipvalue=5.0)
    Loss = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(epochs):
        print("epoch:==========================", epoch)

        if epoch < std:
            loss_list = []
            auc_list = []
            for idx, (data, label) in enumerate(iter(trainloader)):
                with tf.GradientTape() as tape:
                    conv, pred = model(data)
                    loss = Loss(tf.keras.utils.to_categorical(label, num_classes=num_classes), pred)

                gradient = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                loss_list.append(train_loss(loss))
                auc_list.append(train_accuracy(label, pred))
            print("训练集第{}epoch的损失是:{},auc为:{}".format(epoch, cal_avg(loss_list), cal_avg(auc_list)))

            loss_list = []
            auc_list = []
            for idx, (data, label) in enumerate(iter(testloader)):
                with tf.GradientTape() as tape:
                    conv, pred = model(data)
                    loss = Loss(tf.keras.utils.to_categorical(label, num_classes=num_classes), pred)

                loss_list.append(train_loss(loss))
                auc_list.append(train_accuracy(label, pred))
            print("测试集第{}epoch的损失是:{},auc为:{}".format(epoch, cal_avg(loss_list), cal_avg(auc_list)))

        else:  # 第十轮之后开始使用cam自监督

            if len(cam_list) != 0:  # 这时候cam里什么都没有
                temp = []
                for d, c in zip(datas, cam_list):
                    temp.append(core_deal(d, c))
                    # temp.append(final)
                trainloader = to_loader(temp, labels)  # 经过cam处理的新数据
                cam_list = []  # 将之前的cam清空
                print("data has changed now and the camlist is :", len(cam_list))

            loss_list = []
            auc_list = []
            for (data, label) in (iter(trainloader)):
                with tf.GradientTape() as tape:
                    conv, pred = model(data)
                    loss = Loss(tf.keras.utils.to_categorical(label, num_classes=num_classes), pred)

                gradient = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                loss_list.append(train_loss(loss))
                auc_list.append(train_accuracy(label, pred))
            print("训练集第{}epoch的损失是:{},auc为:{}".format(epoch, cal_avg(loss_list), cal_avg(auc_list)))

            for m, n in zip(datas, labels):
                cam = gradcam(model, tf.expand_dims(m, axis=0), tf.expand_dims(n, axis=0))
                cam_list.append(cam)  # datas里的数据和cam一一对应，
            print("The new cams have been got")

            loss_list = []
            auc_list = []
            for idx, (data, label) in enumerate(iter(testloader)):
                with tf.GradientTape() as tape:
                    conv, pred = model(data)
                    loss = Loss(tf.keras.utils.to_categorical(label, num_classes=num_classes), pred)

                loss_list.append(train_loss(loss))
                auc_list.append(train_accuracy(label, pred))
            print("测试集第{}epoch的损失是:{},auc为:{}".format(epoch, cal_avg(loss_list), cal_avg(auc_list)))

    return model


def gradcam(model, inputs, index=None):
    # print("INPUT",inputs.shape,index.shape)
    with tf.GradientTape() as tape:
        feature, prob = model(inputs, training=False)
        # print("FEATURE",feature.shape,prob.shape)
        if not index:
            index = tf.argmax(tf.squeeze(prob))
            # print("INDEX",index)

        target = tf.Variable(tf.zeros(prob.shape[1]))
        target[index].assign(1.0)
        target = tf.reduce_sum(target * prob)

    grads = tape.gradient(target, feature)
    # print("GRADIENT",grads.shape)
    weights = tf.reduce_mean(grads, (0, 1, 2))
    # print("WEIGHT",weights.shape)
    feature = tf.reduce_mean(weights * feature, (0, 3))

    return cv2.resize(feature.numpy(), (width, length))


def main():
    epochs = 10

    data = np.load(project_data_path + font + "data.npy")
    label = np.load(project_data_path + font + "label.npy")

    print(type(data), data.shape, type(label), label.shape)
    trainx, trainy, testx, testy = split_data_multi_channel(data, label)
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)

    trainloader = to_loader(trainx, trainy)
    testloader = to_loader(testx, testy)

    model = VGG2D_Net()
    modeled = train(trainloader, testloader, model)

    notify.send_log(("1612085779@qq.com", "foe3305@163.com"))


if __name__ == '__main__': main()
