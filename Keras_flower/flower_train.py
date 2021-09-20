# -*- coding: utf-8 -*-
"""
版本: 01.24. 22：00 加入了混淆矩阵
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
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import cv2


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
    label_list = [x for x in os.listdir(all_data_path) if os.path.isdir(all_data_path + font + x)]

    label_dic={}
    for i in range(len(label_list)):
        label_dic[label_list[i]]=i

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

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32),label_dic  # np.float32是类型 后面两个变量是没有进行np.asarray


def a_img_preprocess(img_path, w=224, h=224):
    """
    加载一个图片并进行预处理，格式转换等
    :param img_path: 图片文件名
           target_size: 要加载图片的缩放大小
                        之后在image包里面是一个tuple元组类型(w,h)
    :return: 预处理过的图像文件
    """
    img = image.load_img(img_path, target_size=(w, h))
    img = image.img_to_array(img)  # 转换成数组形式
    # print(img.shape)
    '''
    img = np.expand_dims(img, axis=0)  # 为图片增加一维batchsize，直接设置为1
    img = preprocess_input(img)  # 对图像进行标准化
    '''

    return img


def save_pic(img_out_path, superimopsed_img):
    cv2.imwrite(img_out_path, superimopsed_img)


def rotate_img(img):
    """
    随机旋转图像
    :param img: 输入图像
    :return:
    """
    (height, width) = img.shape[:2]

    random_angle = np.random.randint(1, 360)

    center = (height // 2, width // 2)
    matrix = cv2.getRotationMatrix2D(center, random_angle, 1)
    # 旋转图像
    rotate_img = cv2.warpAffine(img, matrix, (width, height))
    # print(rotate_img.shape)
    return rotate_img


def translate_img(img):
    """
    随机平移图像
    :param img: 输入图像
    :return:
    """
    (height, width) = img.shape[:2]
    # 平移矩阵(浮点数类型)  x_shift +右移 -左移  y_shift -上移 +下移
    x_shift = np.random.randint(-100, 100)
    y_shift = np.random.randint(-100, 100)

    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移图像
    trans_img = cv2.warpAffine(img, matrix, (width, height))

    return trans_img


def enhance_data(data, label):
    """
    数据增强，将图片旋转平移之后，数据量增加了
    :param data:
    :param label:
    :return:
    """
    enhanced_data = []
    enhanced_label = []

    for i in range(len(data)):
        slices = data[i]
        s_label = label[i]

        enhanced_data.append(slices)
        enhanced_data.append(translate_img(rotate_img(slices)))
        enhanced_data.append(translate_img(rotate_img(slices)))

        enhanced_label.append(s_label)
        enhanced_label.append(s_label)
        enhanced_label.append(s_label)

    # 转化格式
    enhanced_data, enhanced_label = np.asarray(enhanced_data, np.float32), np.asarray(enhanced_label, np.int32)
    # 打乱顺序
    enhanced_data, enhanced_label = shuffle_data(enhanced_data, enhanced_label)

    print("data enhanced")

    return np.asarray(enhanced_data, np.float32), np.asarray(enhanced_label, np.int32)


def shuffle_data(data, label):
    """
    随机打乱数据集的函数
    :param data: 打乱前的数据npy
    :param label: 打乱前的标签npy
    :return: data, label
    """
    # 打乱顺序
    data_size = data.shape[0]  # 数据集个数
    arr = np.arange(data_size)  # 生成0到datasize个数
    np.random.shuffle(arr)  # 随机打乱arr数组
    data = data[arr]  # 将data以arr索引重新组合
    label = label[arr]  # 将label以arr索引重新组合

    return np.asarray(data, np.float32), np.asarray(label, np.int32)


def split_data(data, label, ratio=0.8):  # 训练集比例 ratio = 0.8
    """
    随机划分训练集与测试集的数据与标签
    :param data: 打乱前的数据npy
    :param label: 打乱前的标签npy
    :param ratio: 训练集占总数据的比例
    :return: x是数据，y是标签
    """
    # 打乱顺序
    data, label = shuffle_data(data, label)

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

    x_train, x_test, y_train, y_test = split_data(data, label)

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


def plot_confusion_matrix(cm, target_names, normalize=True, title='Confusion matrix',
                          picpath='./Confusion_matrix.jpg', show=False, cmap=plt.cm.Greens ):
    """
    :param cm:混淆矩阵
    :param target_names:分类表
    :param normalize:
    :param title:图片内题目
    :param picpath:画图路径
    :param show:是否展示图片
    :param cmap: cmap=plt.cm.Greens这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
    :return:
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    # 这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig(picpath, dpi=350)

    if show:
        plt.show()


def plot_confuse(model, test_data, test_label, label_setting, picpath='./Confusion_Matrix.jpg', show=False):
    """
    显示混淆矩阵
    :param model: 训练结束之后的Keras模型
    :param test_data:测试数据
    :param test_label:测试标签（这里用的是One——hot向量）
    :param label_setting:标签设置的字典
    :return:
    """
    predictions = model.predict_classes(test_data, batch_size=30)
    truelabel = test_label.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)

    # labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
    labels = []
    inverse_setting_dic = {}

    for key, val in label_setting.items():
        inverse_setting_dic[val] = key

    for i in range(len(inverse_setting_dic)):
        # 假设最多有10个类
        if inverse_setting_dic[i] is not None:
            labels.append(inverse_setting_dic[i])
        else:
            break

    plot_confusion_matrix(conf_mat, target_names=labels, normalize=False, title='Confusion Matrix', picpath=picpath,
                          show=show)


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


def begin(project_data_path, all_data_path, model_h5name, reprocess_data=True, batch_size=100, epochs=15, l=224, w=224,
          c=3, target=5, font='/'):
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
    # 读取数据
    if reprocess_data:
        data, label,label_dic = processing_data(all_data_path, l, w, font=font)
        np.save(project_data_path + font + "data.npy", data)
        np.save(project_data_path + font + "label.npy", label)
    else:
        data = np.load(project_data_path + font + "data.npy")
        label = np.load(project_data_path + font + "label.npy")
        label_dic ={"tulips":0,"sunflowers":1,"roses":2,"dandelion":3,"daisy":4}

    # 数据增强
    data, label = enhance_data(data, label)

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
    history = model.fit(x_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test_hot))

    # 保存训练结果+画图呈现训练效果
    model_path = project_data_path + font + model_h5name
    model.save(model_path)

    picpath = project_data_path + font + 'training result.jpg'
    draw_training_summary(history, picpath)
    plot_confuse(model, x_test, y_test_hot, label_dic, picpath='./Confusion_Matrix.jpg', show=False)


font = '/'
# 定义超参数
l = 224
w = 224
c = 3  # 颜色
target = 5  # 花的种类数量

# 定义训练参数
batch_size=100
epochs=2
'''
# remote host
# 项目数据位置
project_data_path = '/home/zty/Insight'
# 所有图片的总路径(目录)
all_data_path = '/home/zty/flower'
# 训练后模型的位置
model_h5name = 'five_flowers_categorical_vgg16.h5'
# local host
# 项目数据位置
project_data_path = '/Users/zhangtianyi/Study/cam_flowers'
# 所有图片的总路径(目录)
all_data_path = '/Users/zhangtianyi/Study/cam_flowers/train_data'
# 训练后模型的位置
model_h5name = 'five_flowers_categorical_vgg16.h5'
'''

# local host
# 项目数据位置
project_data_path = '/Users/zhangtianyi/Study/cam_flowers'
# 所有图片的总路径(目录)
all_data_path = '/Users/zhangtianyi/Study/cam_flowers/train_data'
# 训练后模型的位置
model_h5name = 'five_flowers_categorical_vgg16.h5'


model_path = project_data_path + font + model_h5name

# 测试图像保存位置
plot_pic_path = project_data_path + font + 'resultphoto.jpg'
# begin(project_data_path, all_data_path, model_h5name, True, 15, l, w, c, font='/')

test_pic_path = all_data_path + font + 'daisy/100080576_f52e8ee070_n.jpg'

'''
slices = a_img_preprocess(test_pic_path, w=224, h=224)
new_img = translate_img(rotate_img(slices))
save_pic(plot_pic_path, new_img)
'''

begin(project_data_path, all_data_path, model_h5name, reprocess_data=False)