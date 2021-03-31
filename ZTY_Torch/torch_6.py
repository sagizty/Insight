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

# 划分训练集，测试集 DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

bs = 64
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


# 对于每一个batch，进行模块化处理
def loss_batch(model, loss_func, xb, yb, opt=None):
    # 如果有opt则是训练过程，需要迭代，没有则是验证过程，只需要计算loss就行
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


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


from torch import optim, nn
import torch.nn.functional as F


# 构建模型
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


class Lambda(nn.Module):  # Lambda will create a layer that we can then use when defining a network with Sequential.
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


seq_CNN_model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)


def preprocess(x, y):  # 将input data转换为(-1, 1, 28, 28)，而label y不变
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)  # 把数据移动到device上


class WrappedDataLoader:  # 对dataloader进行处理，增加数据换功能

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


# 看cuda是否可用
print(torch.cuda.is_available())  # True是可用cuda

# 建立device object
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# DataLoader包上格式转换
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

print("testing CNN model ")
lr = 0.1
epochs = 3
loss_func = F.cross_entropy

model = Mnist_CNN().to(dev)

# net = nn.DataParallel(net) 单机多卡
# model = model.DataParallel(model)

# model = seq_CNN_model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Momentum is a variation on stochastic gradient descent
# that takes previous updates into account as well and generally leads to faster training.

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
print("\n\n\n\n\n\n\n")
'''
# 冻结某一层而其他不变
model.fc1.weight.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
# 
# compute loss 
# loss.backward()
# optmizer.step()

# 解冻
model.fc1.weight.requires_grad = True
optimizer.add_param_group({'params': model.fc1.parameters()})

链接：https://www.zhihu.com/question/311095447/answer/589307812
来源：知乎
'''
for name in list(model.named_modules()):
    print(name)

print("start transfer learning")
new_model = Mnist_CNN()
new_model.conv1 = model.conv1  # 迁移这一层过来

new_model.conv1.weight.requires_grad = False

new_model.to(dev)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, new_model.parameters()), lr=lr, momentum=0.9)
fit(epochs, new_model, loss_func, optimizer, train_dl, valid_dl)

print("unfreze")
# 解冻
new_model.conv1.weight.requires_grad = True
# optimizer.add_param_group({'params': new_model.conv1.parameters()})
optimizer = optim.SGD(filter(lambda p: p.requires_grad, new_model.parameters()), lr=lr, momentum=0.9)  # 这样也是重新优化

new_model.to(dev)
new_model = nn.DataParallel(new_model)  # 单机多卡

fit(epochs, new_model, loss_func, optimizer, train_dl, valid_dl)
