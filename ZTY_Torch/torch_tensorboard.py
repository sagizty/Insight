# imports
import shutil
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_myResNet

# 后续计算采用gpu加速
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
                                             download=True,
                                             train=True,
                                             transform=transform)

testset = torchvision.datasets.FashionMNIST('./data',
                                            download=True,
                                            train=False,
                                            transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
# tensorboard部分 建立tensorboard服务器，将要画的图丢上去
'''


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
path = './runs/fashion_mnist_experiment_1'
if os.path.exists(path):
    del_file(path)  # 每次开始的时候都先清空一次

writer = SummaryWriter(path)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

# 在命令行同文件夹跑tensorboard --logdir=/runs --host=10.201.10.16 --port=7777


'''
# 展示模型结构画图
'''
writer.add_graph(net, images)
writer.close()

'''
# 4. Adding a “Projector” to TensorBoard
'''


# We can visualize the lower dimensional representation of higher dimensional data via the add_embedding method
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                     metadata=class_labels,
                     label_img=images.unsqueeze(1))
writer.close()

'''
5. Tracking model training with TensorBoard

In the previous example, we simply printed the model’s running loss every 2000 iterations. 
Now, we’ll instead log the running loss to TensorBoard, along with a view into the predictions the model is making 
via the plot_classes_preds function
'''


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)

    # 转换到cpu上，做npy处理
    preds_tensor = preds_tensor.cpu()

    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # 输入一波数据,只要前4个测试
    if images.shape[0] > 4:
        images = images[0:4]

    batch_size = images.shape[0]

    preds, probs = images_to_probs(net, images)

    # 转换到cpu上做npy，matplotlib处理
    images = images.cpu()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])

        matplotlib_imshow(images[idx], one_channel=True)

        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


# 自定义训练
def fit(model, trainloader, testloader, criterion, optimizer, epochs=2, device='cpu', writer=None):
    for epoch in range(epochs):  # loop over the dataset multiple times

        print("in epoch:", epoch + 1)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # gpu上训练，把数据搬到gpu上
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))

                if writer is not None:  # 记录内容给tensorboard
                    # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / 20,
                                      epoch * len(trainloader) + i)

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    writer.add_figure('predictions vs. actuals',
                                      plot_classes_preds(net, inputs, labels),
                                      global_step=epoch * len(trainloader) + i)

            # empty running loss
            running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    print('Finished Training')


net.to(device)
fit(net, trainloader, testloader, criterion, optimizer, epochs=7, device=device, writer=writer)

'''
writing results to TensorBoard every 1000 batches instead of printing to console
this is done using the add_scalar function.
'''


# 汇总预测概率与内容
def check_performance_result(testloader,net):
    net.to('cpu')  # test at cpu

    class_probs = []
    class_preds = []
    with torch.no_grad():

        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    print(len(test_probs))
    return test_probs, test_preds


# 定义并处理pr曲线数据
def add_pr_curve_tensorboard(writer, class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


# plot the pr curves of all classes
test_probs, test_preds = check_performance_result(testloader,net)

for i in range(len(classes)):
    add_pr_curve_tensorboard(writer, i, test_probs, test_preds)


# you’ll see that on some classes the model has nearly 100% “area under the curve”
