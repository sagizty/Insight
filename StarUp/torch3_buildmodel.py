"""
Third course is about building models and basic knowledge of objects

https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''
basic layers
'''

input_image = torch.rand(2, 3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=3 * 28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

hidden2 = nn.LayerNorm(20)(hidden1)

print(f"After Norm: {hidden2}")


seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Softmax(dim=1)
)
input_image = torch.rand(2,3,28,28)
logits = seq_modules(input_image)
print(logits)

'''
面向对象基本知识

class human(object):

    def __init__(self,birth_year,gender,name):
        self.birth_year=birth_year
        self.gender=gender
        self.name=name

    def gaiming(self,new_name):
        self.name=new_name

    def ask_age(self,this_year):
        print('the age is:',this_year-self.birth_year)

    def change_gender(self):
        self.gender=self.gender *-1 +1

# 实例化一个具体的对象
cjwddcm=human(2000,1,'xzh')
# 读取属性
print(cjwddcm.gender)
# 使用方法
cjwddcm.ask_age(2022)
cjwddcm.change_gender()
# 读取属性
print(cjwddcm.gender)


class master_student(human):
    def __init__(self,birth_year,gender,name,major):
        super().__init__(birth_year,gender,name)

    def change_gender(self):
        print('no way')

wo=master_student(1999,0,'zty','bmds')
print(wo)
print(wo.gender)
wo.change_gender()
print(wo.gender)
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 框架化的模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

input_image = torch.rand(2, 3, 28, 28)
y = model(input_image)

print(y.shape)
print('confidence:', y)
print('predictation:', y.argmax(1))



