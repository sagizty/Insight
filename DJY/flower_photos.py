import tensorflow as tf
import random
import pathlib
# import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, Sequential, metrics, layers


def load_and_preprocess_image(path):
    # 图片的预处理:读取，解码，重构，标准化
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0

    return image

def ds_preprocess(dataset):
    # 对tf数据集预处理：打散、分批、重复
    dataset = dataset.shuffle(1000).batch(100)
    return dataset

def load_data(data_root_orig):
    '''
    加载、划分、预处理数据集
    :param data_root_orig: 输入文件夹路径
    :return: 返回训练和测试的tf数据集
    '''

    # 第一，得到图片路径和对应标签的列表
    data_root = pathlib.Path(data_root_orig)
    all_image_paths = list(data_root.glob('*/*'))
    # print(all_image_paths)
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    # print(label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    # print(label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    # print(len(all_image_labels))
    # print(len(all_image_paths))

    # 第二，依据路径将图片和标签格式化并转化为张量
    all_x = [load_and_preprocess_image(path) for path in all_image_paths]
    all_x = tf.convert_to_tensor(all_x, dtype=tf.float32)
    all_y = tf.convert_to_tensor(all_image_labels, dtype=tf.int32)
    all_y = tf.one_hot(all_y, depth=5)
    print(all_x.shape, all_y.shape)

    # 第三，构建数据集
    all_dataset = tf.data.Dataset.from_tensor_slices((all_x, all_y))
    print(all_dataset)

    # 第四，对数据集预处理
    all_dataset = all_dataset.shuffle(3670).repeat(20)

    # 第五，划分数据集，如果是数组，也可以在fit函数中划分
    test_prop = 0.2
    test_size = int(test_prop * len(all_image_paths))
    train_ds = all_dataset.skip(test_size)
    test_ds = all_dataset.take(test_size)

    test_ds = ds_preprocess(test_ds)
    train_ds = ds_preprocess(train_ds)

    return train_ds, test_ds

# vgg = tf.keras.applications.VGG19(
#         include_top=False, weights='imagenet', input_tensor=None,
#         input_shape=None, pooling=None, classes=1000,
#         classifier_activation='softmax'
#     )
'''    net = Sequential([
        vgg,
        tf.keras.layers.Flatten(), # 这里有问题
        tf.keras.layers.Dense(256, activation=tf.nn.relu).build(input_shape=(None, 25088)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu).build(input_shape=(None, 256)),
        tf.keras.layers.Dense(5).build(input_shape=(None, 128)),
        ])
        '''


def train(epochs, data_root_orig):
    train_ds, test_ds = load_data(data_root_orig)
    print(train_ds, test_ds)
    net = tf.keras.applications.VGG19(
        include_top=False, weights='imagenet', pooling='avg'
    )# 迁移学习
    # 不加全连接时输出7,7,512
    # 加全连接层时输出1000
    net.trainable = False
    network = Sequential([
        net,
        # layers.GlobalAveragePooling2D(),# 可用vgg中pooling=avg代替
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dropout(rate=0.2),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dropout(rate=0.2),
        layers.Dense(5, activation=tf.nn.softmax),
    ])
    network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
    history = network.fit(train_ds, epochs=epochs, validation_data=test_ds)
    network.evaluate(test_ds)
    # 测试vgg网络的输出
    # print(net.summary())
    # x = tf.random.normal([4, 224, 224, 3])
    # out = network(x)
    # print(out.shape)

if __name__ == '__main__':
    # file_path = './flower_photos'
    file_path = '/home/FLOWER-project/train_data'
    train(epochs=50, data_root_orig=file_path)
    # help(tf.losses.CategoricalCrossentropy)


