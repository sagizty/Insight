"""
First course is about nnumpy, torch.tensor

https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html
"""

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.randn(2, 3, 4)
print(f"all row: {tensor[:]}\n")

print(f"2nd element of first dim: {tensor[1]}\n")

print(f"first element of 2 dim of all group: {tensor[:, 0]}\n")

print(f"Last element of 2 dim of all group: {tensor[:, -1]}\n")

print(f"Last dim of all group A: {tensor[:, :, -1]}\n")

print(f"Last dim of all group B: {tensor[..., -1]}\n")

tensor[:, 1] = 0
print(tensor.shape)

tensor2 = tensor.unsqueeze(0)

tensor3 = tensor2.squeeze(0)
print(tensor.shape)
print(tensor2.shape)
print(tensor3.shape,'\n')


tensor2 = tensor.unsqueeze(-1)

tensor3 = tensor2.squeeze(-1)

print(tensor.shape)
print(tensor2.shape)
print(tensor3.shape)


t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1.shape)
t2 = torch.cat([tensor, tensor, tensor], dim=1)
print(t2.shape)
t3 = torch.cat([tensor, tensor, tensor], dim=2)
print(t3.shape)


tensor = torch.randn(2, 3)
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1,'\n',y2)

y3 = torch.rand_like(y1)
print(y3)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)  # 实现in-place操作，替换掉原本y3

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


agg = tensor.sum()
print('agg',agg)
agg_item = agg.item()
print(agg_item, type(agg_item))


print(f"{tensor} \n")
t2=tensor.add(5)
print(t2)
tensor.add_(5)
print(tensor)



t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
