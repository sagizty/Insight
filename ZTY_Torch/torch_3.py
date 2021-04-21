import torch
import torchvision
import torchvision.transforms as transforms

# create dataloadder
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()  # 从此每次迭代出来的就是（images, labels）

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, 5)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        # padding_mode='zeros')
        self.pool = nn.MaxPool2d(2, 2)
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(7, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def training(trainloader, net, optimizer, PATH):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # inputs, labels = data[0].to(device), data[1].to(device)  # gpu mode

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    # saving model
    torch.save(net.state_dict(), PATH)  # 只保存模型参数


# train
PATH = './cifar_net.pth'
# training(trainloader,net,optimizer,PATH)


# loading model
net = Net()  # 意思是需要加载一个空模型之后再把训练好的参数值倒进去
net.load_state_dict(torch.load(PATH))
print("model loaded with state_dict")

# 对于保存和加载整个模型的情况：
# torch.save(model, PATH)
# model = torch.load(PATH)


# 测试一组数据预测效果
dataiter = iter(testloader)
images, labels = dataiter.next()
# print images
imshow(torchvision.utils.make_grid(images))  # make_grid 网格状安排图像tensor
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# predict
outputs = net(images)

print(outputs)  # 预测各类几率的结果向量组 tensor
maxnum, predicted = torch.max(outputs, dim=1)  # dim=1表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值
# torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）。
# 通常写_, predicted = torch.max(outputs, dim=1) 只取最大位置idx
print(maxnum)  # 最大的数值
print(predicted)  # 最大的数值出现的位置idx

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# print(test) test = torch.max(outputs)  这个是找最大的一个元素


correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()  # .squeeze()去除size为1的维度，包括行和列,当维度大于等于2时，squeeze()无作用

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

print('\nAccuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))





# transfer the model network to gpu
# use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# net.to(device)
# Remember that you will have to send the inputs and targets at every step to the GPU too, so alter the training process
# inputs, labels = data[0].to(device), data[1].to(device)