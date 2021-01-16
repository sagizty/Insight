"""
GradCAM
版本：1.16.12.30
参考:
    - https://github.com/gorogoroyasu/mnist-Grad-CAM
    - https://github.com/insikk/Grad-CAM-tensorflow/blob/master/gradCAM_tensorflow_VGG16_demo.ipynb
"""

import sys
import time
import os
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# 1 -> training phase, dropout activated
# 0 -> testing phase, dropout not activated
K.set_learning_phase(0) #set learning phase
npy_path = 'npy_data'


def target_category_loss(x, category_index, nb_classes):
    category_index = tf.argmax(category_index, axis=1)
    return tf.multiply(x, K.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam_batch(model, x, y, num_classes, layer_name, target_only_category):
    """
    Args:
       model: tfkeras.models.Model object
       x: ndarray, input image, which should be preprocessed already,
          either batch or one sample
       y: ndarray, grand truth label
       num_classes: int, number of classes
       layer_name: str or list of str, name of the layer that is target of gradcam
       target_only_category: bool

    Returns:
       jetcam: cv2 ColorMap applied heatmap, range: [0-1]
       heatmap: initial heatmap, range: [0-1]
       predicted_cls: predictions

    Note:
        colormap ref: https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#gga9a805d8262bcbe273f16be9ea2055a65ab3f207661ddf74511b002b1acda5ec09
    """
    assert x.shape[0] == y.shape[0]
    assert x.ndim == 4
    assert y.ndim == 2

    h, w = x.shape[1:3]
    batch_size = x.shape[0]

    if target_only_category:
        input_gt_label = tf.keras.layers.Input(shape=(num_classes), name='input_gt_label')
        target_layer = lambda x: target_category_loss(x[0], x[1], num_classes)
        model_output = model.layers[-1].output
        output = tf.keras.layers.Lambda(target_layer, output_shape=target_category_loss_output_shape)([model_output, input_gt_label])
        model = tf.keras.models.Model(inputs=[model.input,input_gt_label], outputs=output)

    conv_layer = model.get_layer(layer_name)
    gradcam_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    with tf.GradientTape() as tape:
        conv_output, predictions = gradcam_model([x, y])
        predicted_cls = np.argmax(predictions, axis=1)
        model_output = tf.reduce_sum(predictions, axis=1)
        grads = tape.gradient(model_output, conv_output)
        # weights = normalize(K.mean(grads, axis=(1, 2))) # grads.shape = (batch, h, w, c) -> (batch, c)
        weights = normalize(tf.reduce_mean(grads, axis=(1, 2))) # grads.shape = (batch, h, w, c) -> (batch, c)

    # w * conv -> (b, h, w, c) -> reduce_mean -> (b, h, w)
    # multiply -> (b, h, w, c)
    # reduce_mean -> (b, h, w)
    cams = tf.einsum('ijkl,il->ijk', conv_output, weights)
    #  <--> cams_np = np.asarray([tf.reduce_sum(tf.multiply(weights[b], conv_output[b]), axis=-1) for b in range(batch_size)])
    # cams.shape == cam_np.shape
    # print(np.count_nonzero(cams.numpy() == cams_np)) == print(cams_np.size)

    jetcams = []
    heatmaps = []
    for n in range(batch_size):
        cam = cams[n].numpy()
        cam = cv2.resize(cam, (w,h))
        # ReLU activation
        cam = np.maximum(cam, 0)
        # [0-1] normalization
        cam = cam / np.max(cam) if np.max(cam) != 0.0 else cam / 1e-10

        heatmaps += [cam]
        # cam -> [0-255] -> colormap -> [0-1]
        jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
        jetcams += [jetcam / 255]

    return jetcams, heatmaps, predicted_cls


def main(args):
    sample_index = args.sample_index
    batch_size = args.batch_size
    save = args.save_plot

    num_classes = 5

    model = tf.keras.models.load_model(f'model/{args.model_name}')
    model.summary()

    layer_names = [l.name for l in model.layers if 'conv' in l.name]
    layer_names = layer_names[::-1][1:3][::-1]
    print('target conv lyaers:', layer_names)

    # _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_test = x_test / 255.
    # x_test = x_test.reshape((-1, 28, 28, 1))
    # y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # 读取numpy数据
    # _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print(x_test.shape)
    # print(y_test.shape)


    x_test = np.load(os.path.join(npy_path, 'x_test.npy'))
    y_test = np.load(os.path.join(npy_path, 'y_test.npy'))
    x_test *= 255
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x = x_test[sample_index: sample_index + batch_size]
    y = y_test[sample_index: sample_index + batch_size]
    print('Input data shape')
    print(f'x: {x.shape}, y: {y.shape}')

    jetcams = []
    graycams = []
    predictions = 0
    for i, l_name in enumerate(layer_names):
        # GradCAM
        jetcam, graycam, prediction = grad_cam_batch(model, x, y, num_classes, l_name, True)
        if i == 0:
            predictions = prediction
        # draw cam on input x
        jetcam = x + jetcam
        # cam -> [0-1]
        max_value = np.max(jetcam, axis=(1,2,3))
        jetcams += [[jetcam[b] / max_value[b] for b in range(batch_size)]]
        # graycam
        grays = []
        for b in range(batch_size):
            gray = cv2.applyColorMap(np.uint8(255 * graycam[b]), cv2.COLORMAP_JET)
            grays += [cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)]
        graycams.append(grays)

    # plot
    w = (1 + len(layer_names) + 1)
    h = batch_size
    batch_gt_labels = np.argmax(y, axis=1)

    fig, ax = plt.subplots(h, w, figsize=(w*1.5, h*2))

    flower_dict = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    for b in range(batch_size):
        # input image
        ax[b, 0].imshow(x[b])
        c = ax[b,0].imshow(x[b].reshape((x[b].shape[0], x[b].shape[1], 3)))
        ax[b,0].set_title(f'gt:{flower_dict[batch_gt_labels[b]]}, pred:{flower_dict[predictions[b]]}', fontsize=11)
        # gradcam
        for i in range(len(layer_names)):
            c = ax[b, i+1].imshow(jetcams[i][b], cmap='jet')
            ax[b, i+1].set_title(layer_names[i])
        # last layer's heatmap
        c = ax[b, -1].imshow(graycams[-1][b],cmap='jet')
        ax[b, -1].set_title(f'{layer_names[i]}\nheatmap only',fontsize=9)

    fig.suptitle('GradCAM MNIST')

    if save:
        Path('pic/svg').mkdir(exist_ok=True)
        Path('pic/png').mkdir(exist_ok=True)
        plt.savefig(f'pic/svg/gradcam_mnist.svg')
        plt.savefig(f'pic/png/gradcam_mnist.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # 定义参数
    sample_index = 15   # 图片采样位置
    batch_size = 3  # 采样个数


    parser = argparse.ArgumentParser(description='GradCAM flower')
    parser.add_argument('--sample-index', '-s', type=int, default=sample_index)
    parser.add_argument('--batch-size', '-b', type=int, default=batch_size)
    parser.add_argument('--save-plot', '-sp', action='store_true')
    parser.add_argument('--model-name', '-m', type=str, default='model.hdf5')
    args = parser.parse_args()

    main(args)
