"""
版本：12 29 00:39
"""

import numpy as np
import os
import glob
from skimage import transform, io
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
#big server 
project_data_path = '/home/FLOWER-project/Insight'
'''

font = '/'
# local host
# 项目数据位置
project_data_path = '/Users/zhangtianyi/Study/cam_flowers'


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


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv = tf.keras.Sequential(name='conv')
        # for i in [64,128,256,512,512]:
        for i in [64, 128]:
            self.conv.add(layers.Conv2D(i, 3, strides=(1, 1), padding='same', activation='relu'))
            self.conv.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.linear = tf.keras.Sequential(name='linear')
        self.linear.add(layers.Flatten())
        # for i in range(2):
        #     self.linear.add(layers.Dense(4096,activation='relu'))
        #     self.linear.add(layers.Dropout(0.2))
        self.linear.add(layers.Dense(5))

    def call(self, x):
        conv = self.conv(x)
        pred = tf.nn.softmax(self.linear(conv))

        return conv, pred


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


def train(dataloader, model, epochs):
    optimizer = tf.keras.optimizers.Adam(0.001)
    Loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    for epoch in epochs:
        for idx, (data, label) in enumerate(dataloader):
            with tf.GradientTape() as tape:
                pred = model(data)

                loss = Loss(pred, label)

            gradient = tape.gradient(loss, model.trainable._variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))


def to_loader(data, label, fix=64):
    """
    total=[]
    a=data.shape[0]//fix
    for i in range(0,a+1):
        loader=tf.data.Dataset.from_tensor_slices((data[i*fix:(i+1)*fix],label[i*fix,(i+1)*fix]))
        loader=loader.shuffle(buffer_size=True)
        loader=loader.batch(batch_size=fix)
        loader=iter(loader)
        total.append(loader)
    return loader
    """

    loader = tf.data.Dataset.from_tensor_slices((data, label))
    loader = loader.shuffle(buffer_size=1000)
    loader = loader.batch(batch_size=fix)

    return loader
    # 读取数据


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

    optimizer = tf.keras.optimizers.Adam(0.0005, clipvalue=5.0)
    Loss = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(epochs):

        print("epoch:==========================", epoch)
        for idx, (data, label) in enumerate(iter(trainloader)):
            with tf.GradientTape() as tape:
                conv, pred = model(data)
                loss = Loss(tf.keras.utils.to_categorical(label, num_classes=5), pred)

            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            print(train_loss(loss))
            print(train_accuracy(label, pred))

        for idx, (data, label) in enumerate(iter(testloader)):
            with tf.GradientTape() as tape:
                conv, pred = model(data)

                print(train_accuracy(label, pred))


if __name__ == '__main__': main()
