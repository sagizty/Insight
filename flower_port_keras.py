from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
import cv2
from PIL import Image
import matplotlib.pyplot as plt

'''
参考资料：
1。凭什么相信你，我的CNN模型？（篇一：CAM和Grad-CAM)
https://bindog.github.io/blog/2018/02/10/model-explanation/
2。卷积神经网络的可视化——热力图Grad CAM
https://blog.csdn.net/weixin_44106928/article/details/103323970
3。keras CAM和Grad-cam原理简介与实现
https://blog.csdn.net/MrR1ght/article/details/92799591
4。Tensorflow框架搭建卷积神经网络进行五种花的分类
https://blog.csdn.net/AugustMe/article/details/94166164
5.【keras实战】用Inceptionv3实现五种花的分类
https://blog.csdn.net/m0_37935211/article/details/83003554
'''


# Grand CAM尝试

def load_model_h5(model_file):
    """
    载入原始keras模型文件
    :param model_file: 模型文件，h5类型
    :return: 模型
    """
    model = load_model(model_file)
    model.summary()
    # 这里可以知道每层layer的名字!!!!
    return model


def load_img_preprocess(img_path, w=224, h=224):
    """
    加载图片并进行预处理，格式转换等
    :param img_path: 图片文件名
           target_size: 要加载图片的缩放大小
                        之后在image包里面是一个tuple元组类型(w,h)
    :return: 预处理过的图像文件
    """
    img = image.load_img(img_path, target_size=(w, h))
    img = image.img_to_array(img)  # 转换成数组形式
    img = np.expand_dims(img, axis=0)  # 为图片增加一维batchsize，直接设置为1

    img = preprocess_input(img)  # 对图像进行标准化

    return img


def show_pre(model, img, labels):
    # 输出预测结果，需要根据任务而变化
    print("预测类别：" + labels[int(model.predict_classes(img))])  # 预测结果（预测图片最大分类可能性结果）


def gradient_compute(model, layername, img):
    """
    计算模型最后输出与你的layer的梯度
    并将每个特征图的梯度进行平均
    再将其与卷积层输出相乘
    :param model: 模型
    :param layername: 你想可视化热力的层名
    :param img: 预处理后的图像
    :return:
    卷积层与平均梯度相乘的输出值
    """

    preds = model.predict(img)  # 预测结果（预测图片最大分类可能性结果）
    idx = np.argmax(preds[0])  # 返回预测结果的index索引

    output = model.output[:, idx]  # 获取到我们对应索引的输出张量！！！！
    last_layer = model.get_layer(layername)  # 定位到最后一个卷积层

    grads = K.gradients(output, last_layer.output)[0]  # 获取最后一个卷积之后该结果张量在输出结果上的梯度

    pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 对整张梯度特征图在0， 1， 2维度上进行平均，
    # pooled_grads返回的是一个大小是通道维数的张量！！！！！！

    iterate = K.function([model.input], [pooled_grads, last_layer.output[0]])

    # iterate这里调用keras的backend后端，对一个函数进行实例化，即执行我们前面定义的操作
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # 将其与卷积层输出相乘
    for i in range(pooled_grads.shape[0]):  # 利用一个for循环进行加权
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # 获得conv_layer_output_value是原图尺寸各位置有支持分类的梯度值作为权重的"图"！！！
    return conv_layer_output_value


def plot_heatmap(conv_layer_output_value, img_in_path, img_out_path):  # 将卷积层按梯度获得权重后的各像素位置画热力图 & 叠加
    """
    绘制热力图
    :param conv_layer_output_value: 卷积层输出值
    :param img_in_path: 输入图像的路径
    :param img_out_path: 输出热力图的路径
    :return:
    """
    heatmap = np.mean(conv_layer_output_value, axis=-1)  # 生成像素权重"图"
    heatmap = np.maximum(heatmap, 0)  # 人工relu，实现只要正向支持结果的部分入"图"
    heatmap /= np.max(heatmap)  # Normalize between 0-1 将数值正态化

    img = cv2.imread(img_in_path)  # 原图
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 对热力"图"修改到与原图片一样的尺寸大小
    heatmap = np.uint8(255 * heatmap)  # 对数值进行色域修改赋值，色域在 0-255

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将数值变成颜色
    superimopsed_img = heatmap * 0.4 + img  # 涂上去的热力图颜色深度需要调整，之后进行叠加

    cv2.imwrite(img_out_path, superimopsed_img)  # 在指定路径生成图片


def show_grand_cam_result(test_img_path, model_path, layername, plot_pic_path, labels, w=224, h=224):
    model = load_model_h5(model_path)
    img = load_img_preprocess(test_img_path, w, h)

    show_pre(model, img, labels)
    conv_value = gradient_compute(model, layername, img)
    plot_heatmap(conv_value, test_img_path, plot_pic_path)

    plt.imshow(Image.open(plot_pic_path).convert('RGB'))
    plt.show()


# 定义超参数
w = 224
h = 224
c = 3  # 颜色
labels = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# 数据位置
project_data_path = '/Users/zhangtianyi/Study/cam_flowers'
all_data_path = '/Users/zhangtianyi/Study/cam_flowers/train_data'  # 所有图片的总路径(目录)

#训练后模型的位置
model_h5name = 'five_flowers_categorical_vgg16.h5'
model_path = project_data_path +font + model_h5name


# 最后一个卷积层的名字 通过model.summary()看
layername = r'separable_conv2d_6'


# 测试图片位置，，输出存储cam图片的位置
test_pic_path = all_data_path +font+'1044296388_912143e1d4.jpg'

plot_pic_path = project_data_path +font+'resultpic2.jpg'

# Grand CAM测试
show_grand_cam_result(test_pic_path, model_path, layername, plot_pic_path, labels, w, h)
