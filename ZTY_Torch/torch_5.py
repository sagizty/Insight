# 不使用torch.nn 来构建神经网络，学习神经网络的torch底层

# 加载数据

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
pyplot.show()
print(x_train.shape)  # 检查加载的数据

print("\n\n\n\n\n\n\n")

# 划分训练集，测试集
import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

print("\n\n\n\n\n\n\n")

# define model of an ANN

import math

# 定义 ANN中需要的tensor：weight
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()  # 需要 gradient， _ in PyTorch signifies that the operation is performed in-place.
bias = torch.zeros(10, requires_grad=True)


# define the softmax computing func we need
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):  # basic liner matrix ANN,这个模型的输出是每个目标的分类概率vecto
    return log_softmax(xb @ weights + bias)  # @ stands for the dot product operation


# 以一个64张的batch的数据集合进行测试
bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
print("one preds:", preds[0])
print("preds shape:", preds.shape)
'''
output:

tensor([-2.4723, -2.4401, -2.1336, -2.3609, -2.6171, -2.1957, -1.8676, -2.1560,
        -2.8847, -2.2513], grad_fn=<SelectBackward>) torch.Size([64, 10])
        
# grad_fn=<SelectBackward> 
# the preds tensor contains not only the tensor values, 
but also a gradient function
'''
print("\n\n\n\n\n\n\n")


# loss func
# implement negative log-likelihood to use as the loss function
def nll(input, target):  # 这个是分类的loss，分类的概率结果与目标类标签的loss
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

# testing loss
yb = y_train[0:bs]

print(preds)
print(yb)
print("loss result :", loss_func(preds, yb))
'''
output:

tensor(2.2863, grad_fn=<NegBackward>)

'''


# get result
def accuracy(output_probs, yb):  # out=a batch of pred output tensor , yb= a batch of ground truth label tensor
    preds_idx = torch.argmax(output_probs, dim=1)  # torch.argmax get the position idx of the max item
    # like idx,_ =torch.max(out, dim=1) dim is a request for row(1) or col(0)

    return (preds_idx == yb).float().mean()  # matching acc


print(accuracy(preds, yb))

'''
output:

tensor(0.0469)

'''
print("\n\n\n\n\n\n\n")
'''
We can now run a training loop. For each iteration, we will:

select a mini-batch of data (of size bs)
use the model to make predictions
calculate the loss
loss.backward() updates the gradients of the model, in this case, weights and bias.

loss.backward() adds the gradients to whatever is already stored, rather than replacing them).
'''

# Training loop use SGD

from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()

        # SGD optimise
        with torch.no_grad():
            # We do this within the torch.no_grad() context manager,
            # because we do not want these actions to be recorded for our next calculation of the gradient.
            weights -= weights.grad * lr
            bias -= bias.grad * lr

            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
'''
output:

tensor(0.0816, grad_fn=<NegBackward>) tensor(1.)

'''
print("\n\n\n\n\n\n\n")

# 用torch.nn来替代构造的过程

import torch.nn.functional as F

loss_func = F.cross_entropy


def model(xb):
    return xb @ weights + bias


print(loss_func(model(xb), yb), accuracy(model(xb), yb))
print("\n\n\n\n\n\n\n")

# 修改迭代逻辑

from torch import nn


class Mnist_Logistic(nn.Module):  # 继承nn.Module来构建自己的网络
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


model = Mnist_Logistic()

print("pred;", model(xb))
print(loss_func(model(xb), yb), yb)

'''
# manaul SGD    one step
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
'''


def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()

            with torch.no_grad():  # SGD,torch.no_grad() because we dont want this procedule was record with gard
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()  # clean the gradient


fit()
print(loss_func(model(xb), yb))

'''
 Instead of manually defining and initializing self.weights and self.bias, and calculating xb  
 @ self.weights + self.bias, we will instead use the Pytorch class nn.Linear for a linear layer
'''


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 250)
        self.lin2 = nn.Linear(250, 10)

    def forward(self, xb):  # 因为是单层

        x = self.lin1(xb)
        x = self.lin2(x)

        return x


model = Mnist_Logistic()
print(loss_func(model(xb), yb))

fit()

print(loss_func(model(xb), yb))

print("\n\n\n\n\n\n\n")

from torch import optim

model = Mnist_Logistic()
opt = optim.SGD(model.parameters(), lr=lr)

print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()

        # use opt.step we can do iteration change easily , no need for 'with torch.no_grad():'
        # since it have been included already
        opt.step()
        opt.zero_grad()  # we still need to clear the gradient at the end of each loop step

print(loss_func(model(xb), yb))

"""
DATAset
PyTorch has an abstract Dataset class. A Dataset can be anything that has a __len__ function (called by 
Python’s standard len function) and a __getitem__ function as a way of indexing into it. 
This tutorial walks through a nice example of creating a custom FacialLandmarkDataset class as a subclass of Dataset.

PyTorch’s TensorDataset is a Dataset wrapping tensors. By defining a length and way of indexing, this also 
gives us a way to iterate, index, and slice along the first dimension of a tensor. This will make it easier 
to access both the independent and dependent variables in the same line as we train.
"""
print("\n\n\n\n\n\n\n")

# dataset
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

# in the orthodox way
# xb = x_train[start_i:end_i]
# yb = y_train[start_i:end_i]

# now
# xb,yb = train_ds[i*bs : i*bs+bs]

model = Mnist_Logistic()
opt = optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):

    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]

        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

print("\n\n\n\n\n\n\n")

# data loader  DataLoader gives us each minibatch automatically.

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)  # setting batch size

# taking data into testing loop
for xb, yb in train_dl:
    pred = model(xb)

# redefine the model and the optimizer
model = Mnist_Logistic()
opt = optim.SGD(model.parameters(), lr=lr)

# starting the training loop
for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# out

# tensor(0.0816, grad_fn=<NllLossBackward>)

'''
训练集需要打散
Shuffling the training data is important to prevent correlation between batches and overfitting.

关于验证集：
不需要shuffle，因为反正都是全部跑一轮用来验证效果，此过程的loss和acc并不会对模型有影响，因此不会影响模型收敛
是设置为验证模式，不记录梯度，不迭代。数据其实是没有进入模型的。

'''
print("\n\n\n\n\n\n\n")

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
'''
We’ll use a batch size for the validation set that is twice as large as that for the training set. 
This is because the validation set does not need backpropagation and thus takes less memory 
(it doesn’t need to store the gradients). We take advantage of this to use a larger batch size and 
compute the loss more quickly.
'''
# redefine the model and the optimizer
model = Mnist_Logistic()
opt = optim.SGD(model.parameters(), lr=lr)

# We will calculate and print the validation loss at the end of each epoch.
'''
Note that we always call model.train() before training, and model.eval() before inference, 
because these are used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour 
for these different phases.
'''
for epoch in range(epochs):
    model.train()

    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

'''
out:
0 tensor(0.4252)
1 tensor(0.3379)
'''
print("\n\n\n\n\n\n\n")


# 对于每一个batch，进行模块化处理


def loss_batch(model, loss_func, xb, yb, opt=None):
    # 如果有opt则是训练过程，需要迭代，没有则是验证过程，只需要计算loss就行
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


# 将整个训练验证过程设置为fit
import numpy as np


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):

        # 迭代训练
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        # 验证
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


# create get dataloader func
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


# create get model and optimizer func
def get_model():
    model = Mnist_Logistic()
    opt = optim.SGD(model.parameters(), lr=lr)
    return model, opt


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

'''
来个CNN试验一下这个框架
'''


class Mnist_CNN(nn.Module):  # 首先是继承nn.Module

    def __init__(self):
        super().__init__()  # 继承__init__，之后构建layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):  # 定义数据流
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


print("testing CNN model ")
lr = 0.1
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Momentum is a variation on stochastic gradient descent
# that takes previous updates into account as well and generally leads to faster training.

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

"""
设计自己的层，并且插入到nn.squential搭建的模型中
"""
print("\n\n\n\n\n\n\n")

class Lambda(nn.Module):  # Lambda will create a layer that we can then use when defining a network with Sequential.
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)  # torch.view 相当于numpy中resize（）的功能，但是用法可能不太一样
    # 把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor
    # -1 自动化适应minibatch 1指每个人用一个纬度 28*28是数据


# 用nn.Squentical 构建模型
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

'''
把数据格式转换包到dataloader里面
'''
print("\n\n\n\n\n\n\n")

def preprocess(x, y):
    # 将input data转换为(-1, 1, 28, 28)，而label y不变
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):  # 将迭代规则改写
        batches = iter(self.dl)  # 迭代一次，获得的为batches

        for b in batches:
            # 把这一轮的材料 传入func中做变换
            yield (self.func(*b))  # *b 是指：因为输入的b其实是一个data 一个label，* 指分开传入，而不是作为一个元组传入 func
            # yield 生成一轮


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)  # 以batch获得data 和 label，包上了之前dataset获得的data 和 label

# 包上格式转换
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
'''
replace nn.AvgPool2d with nn.AdaptiveAvgPool2d, which allows us to define the size of the output tensor we want, 
rather than the input tensor we have. As a result, our model will work with any size input.
'''
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

"""
学习搬到gpu上。数据和模型搬上去就行，其他的操作其实是一样的

搬数据本质上是每次读到的数据放到dev上

搬数据的过程可用warp到dataloader的构建中去，这样dataloader实现的时候调取的数据就到dev上了

搬模型则只需要model.to(dev)就可以把模型搬到dev上
"""
print("\n\n\n\n\n\n\n")

# 看cuda是否可用
print(torch.cuda.is_available())  # True是可用cuda

# 建立device object
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Let’s update preprocess to move batches to the GPU:
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)  # 把数据移动到device上


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

# move our model to the GPU
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # 迭代器还是一样的

fit(epochs, model, loss_func, opt, train_dl, valid_dl)  # 训练也是一样的
