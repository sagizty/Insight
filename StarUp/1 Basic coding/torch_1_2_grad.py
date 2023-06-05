import torch, torchvision

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

print(a)

Q = 3 * a ** 3 - b ** 2

print(Q)
print(a.grad)  # none

# We need to explicitly pass a gradient argument in Q.backward() because it is a vector.
# gradient is a tensor of the same shape as Q, and it represents the gradient of Q w.r.t. itself, i.e.
# dQ/dQ =1
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(Q)

# 等价于 aggregate Q into a scalar and call backward implicitly
# Q.sum().backward()
# 如果直接Q.backward() 会有RuntimeError: grad can be implicitly created only for scalar outputs

# Gradients are now deposited in a.grad and b.grad
print(a.grad)

print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)


# 对于单个tensor的求导过程
import math
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# optim，输入目标对象，设置进行迭代反传调整的对象与方法
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# in your training loop:
optim.zero_grad()  # zero the gradient buffers

# forward
# 1。predict
pred = model(data)
# 2。check loss
loss = (pred - labels).sum()

# backward
# 1。autogard
loss.backward()
# 2。refreash
optim.step()


# 迁移学习部分
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# 换层，新层的参数默认是requires_grad = True
model.fc = torch.nn.Linear(512, 10)  # 设置输出为10个类

# Optimize only the classifier: model.fc.parameters()，有个小疑问，这里如果把对象换成model.parameters()那么是不是等价的，
# 因为除了fc layer的都被冻住了，设置了不计算保留梯度？
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
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
# 再来一次
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 10)  # 随机假设结果为10个类

# in your training loop:
optim.zero_grad()  # zero the gradient buffers
# forward
pred = model(data)
# loss
loss = (pred - labels).sum()
# gard
loss.backward()
# backward
optim.step()
