'''
CNN
版本：1.16.12.30
代码参考：
    - https://tensorflow.google.cn/tutorials/images/cnn?hl=zh_cn
    - https://github.com/KeishiIshihara/keras-gradcam-mnist

模型的训练精度只有60多，但gradcam效果比较好
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import pathlib
import datetime

# 1. 定义超参
npy_path = 'npy_data'   # 预处理好的数据集位置
input_shape = (100, 100, 3)     # 输入图片的大小（w, h, c）
epochs = 10     # 训练次数
num_classes = 5     # 分类个数

# 2. 读取数据
train_images = np.load(os.path.join(npy_path, 'x_train.npy'))
train_labels = np.load(os.path.join(npy_path, 'y_train.npy'))
test_images = np.load(os.path.join(npy_path, 'x_test.npy'))
test_labels = np.load(os.path.join(npy_path, 'y_test.npy'))
train_images *= 255
test_images *= 255

# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
# print(train_labels.shape)

# 3. 输入模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes))    # 模型的最后一层要将维度降到和分类个数一致
model.summary()     # 命令行查看模型结构

# 4. 配置tensorboard，用于监控训练过程
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 5. 编译模型
'''
参数内容：
model.compile(optimizer='优化器',
              loss = '损失函数',
              metrics = ["准确率"])
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 6. 训练模型
history = model.fit(train_images, train_labels, epochs=epochs,
                    validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# 7. 打印运行结果
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)  # 获取最终loss和准确率
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# 8. 将模型保存在model/model.hdf5下
pathlib.Path('model').mkdir(exist_ok=True)
model.save('model/model.hdf5')

# 训练完成后可以通过在命令行中输入 tensorboard --logdir=./logs 启动tensorboard，然后在 http://localhost:6006/ 查看模型信息


