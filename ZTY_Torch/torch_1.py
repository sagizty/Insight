import torch
import numpy as np


data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)
np_array = np.array(data)

x_np = torch.from_numpy(np_array)
print(x_np)
y_np = torch.ones_like(x_np)
print(y_np)

shape=(2,3,)  # 与(2,3,)一样
z_np = torch.rand(shape)
print(z_np)

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

print(tensor, "\n")
tensor.add_(5)
print(tensor)

print("\n")

tensor1 = tensor.add(5)
print(tensor)
print(tensor1)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

print("\n")

n = np.ones(5)
t = torch.from_numpy(n)

b = np.add(n, 2)
np.add(n, 1, out=n)  # out=n 使得变化是在n上的了，而这种实现在pytorch 的tensor里面用后缀动作实现，如t.add_（x）


print(f"t: {t}")
print(f"b: {b}")
print(f"n: {n}")
