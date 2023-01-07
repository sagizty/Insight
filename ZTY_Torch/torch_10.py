"""
Version: Jan 7 2023

use torch tensor to build a CNN network
with only tensor
without autograd and everything

author Tianyi Zhang

"""

import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm


def draw(train_res, val_res):
    plt.plot(train_res, 'r', label='Training')
    plt.plot(val_res, 'b', label='Testing')
    plt.ylabel('loss (MSE)')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.title('CNN fitting curve')
    plt.show()


def minist_preprocess(input_batch_img, batch_label, num_classes=10, flattern=True):
    # B,1,28,28
    if flattern:
        # [B,1,28,28] -> [B,1*28*28]
        inputs = input_batch_img.view(-1, 784)
    else:
        # [B,1,28,28] -> [B,3,28,28]
        inputs = torch.cat((input_batch_img, input_batch_img, input_batch_img), 1)
        inputs = inputs.view(-1, 3, 28, 28)

    labels = torch.zeros((batch_label.size(0), num_classes))
    for i in range(batch_label.size(0)):
        labels[i, batch_label[i]] = 1

    # no need for torch grad
    return inputs.detach(), labels.detach()


class Reshape:
    def __init__(self, input_size, output_size):
        # Store the input and output sizes
        self.input_size = list(input_size) if type(input_size) is not int else [input_size,]
        self.output_size = list(output_size) if type(output_size) is not int else [output_size,]

    def forward(self, x):
        # Store the original shape of the input tensor
        list = [x.shape[0],]
        list.extend(self.input_size)
        self.original_shape = list

        list = [x.shape[0],]
        list.extend(self.output_size)

        # Reshape the input tensor
        return torch.reshape(x, list)

    def backward(self, grad_output):
        # Compute the gradient of the reshaped tensor with respect to the original tensor
        grad_input = torch.reshape(grad_output, self.original_shape)

        return grad_input


"""
a = Reshape((3, 224, 224),(672, 224))

x = torch.randn(2, 3, 224, 224)
y = torch.randn(2, 672, 224)

z = a.forward(x)
print(z.shape)
z= a.backward(y)
print(z.shape)
"""


class MaxPool:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        self.input = x

        batch_size, channels, height, width = self.input.shape
        outputs = torch.zeros(batch_size, channels,
                              (height + 2 * self.padding - self.kernel_size) // self.stride + 1,
                              (width + 2 * self.padding - self.kernel_size) // self.stride + 1)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(0, height + 2 * self.padding - self.kernel_size + 1, self.stride):
                    for w in range(0, width + 2 * self.padding - self.kernel_size + 1, self.stride):
                        outputs[b, c, h // self.stride, w // self.stride] = \
                            self.input[b, c, h:h+self.kernel_size, w:w+self.kernel_size].max()

        return outputs

    def backward(self, grad_output):
        grad_input = torch.zeros(self.input.shape)

        batch_size, channels, height, width = self.input.shape
        for b in range(batch_size):
            for c in range(channels):
                for h in range(0, height + 2 * self.padding - self.kernel_size + 1, self.stride):
                    for w in range(0, width + 2 * self.padding - self.kernel_size + 1, self.stride):
                        window = self.input[b, c, h:h+self.kernel_size, w:w+self.kernel_size]
                        max_index = window.argmax()
                        grad_input[b, c, h + max_index // self.kernel_size, w + max_index % self.kernel_size] \
                            = grad_output[b, c, h // self.stride, w // self.stride]

        return grad_input


'''
pooling = MaxPool(kernel_size=3, stride=1, padding=0)
x = torch.randn([2, 3, 30, 30])
y = pooling.forward(x)
print(y.shape)
err_x = torch.randn([2, 3, 28, 28])
y = pooling.backward(err_x)
print(y.shape)
'''

class AvgPool:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        self.input = x
        # Calculate the size of the output tensor
        batch_size, channels, in_height, in_width = self.input.size()
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Create an output tensor filled with zeros
        output = torch.zeros(batch_size, channels, out_height, out_width, dtype=self.input.dtype,
                             device=self.input.device)

        # Zero-pad the input tensor if necessary
        if self.padding > 0:
            self.input = torch.nn.functional.pad(self.input, (self.padding, self.padding, self.padding, self.padding))
            # Create a padded tensor filled with zeros
            padded = torch.zeros(batch_size, channels, in_height + 2 * padding, in_width + 2 * padding,
                                 dtype=self.input.dtype, device=self.input.device)
            # Copy the input tensor to the center of the padded tensor
            padded[:, :, padding:padding + height, padding:padding + width] = self.input
        else:
            padded = self.input

        # Loop over the batch and channels dimensions
        for b in range(batch_size):
            for c in range(channels):
                # Loop over the output height and width dimensions
                for i in range(out_height):
                    for j in range(out_width):
                        # Calculate the start and end indices of the current kernel
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Average the values in the current kernel and store them in the output tensor
                        output[b, c, i, j] = torch.mean(padded[b, c, h_start:h_end, w_start:w_end])

        return output

    def backward(self, grad_output):
        # Calculate the size of the input tensor
        batch_size, channels, in_height, in_width = self.input.size()
        out_height, out_width = grad_output.size()[-2:]

        # Create a gradient tensor filled with zeros
        grad_input = torch.zeros_like(self.input)

        # Loop over the batch and channels dimensions
        for b in range(batch_size):
            for c in range(channels):
                # Loop over the output height and width dimensions
                for i in range(out_height):
                    for j in range(out_width):
                        # Calculate the start and end indices of the current kernel
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Average the gradient values in the current kernel and store them in the gradient tensor
                        grad_input[b, c, h_start:h_end, w_start:w_end] = grad_output[b, c, i, j] / (
                                    self.kernel_size * self.kernel_size)

        return grad_input


'''
pooling = AvgPool(kernel_size=3, stride=1, padding=0)
x = torch.randn([2, 3, 30, 30])
y = pooling.forward(x)
print(y.shape)
err_x = torch.randn([2, 3, 28, 28])
y = pooling.backward(err_x)
print(y.shape)
'''


class FC:
    def __init__(self, input_size, output_size):
        # no need for torch grad
        self.weight = torch.randn(input_size, output_size, requires_grad=False)
        self.bias = torch.randn(output_size, requires_grad=False)

        self.params = {"weight": self.weight, "bias": self.bias}
        self.params_grad = {"weight": None, "bias": None}

    def forward(self, x):
        self.input = x
        return torch.matmul(self.input, self.weight) + self.bias.expand(self.input.size(0), -1)

    def backward(self, grad_output):
        grad_input = torch.matmul(grad_output, self.weight.t())
        grad_weight = torch.matmul(self.input.t(), grad_output)
        grad_bias = grad_output.sum(0)
        # update grad for pamrameter
        self.params_grad = {"weight": grad_weight, "bias": grad_bias}
        return grad_input


'''
mod = FC(10,24)
x = torch.randn([2,10])
err_x = torch.randn([2,24])

y = mod.forward(x)
print(y)

y = mod.backward(err_x)
print(y)
'''


class ReLU:
    def forward(self, x):
        self.input = x
        return torch.clamp(self.input, min=0)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


'''
x = torch.randn([2,10])
err_x=torch.randn([2,10])
ac=ReLU()
y=ac.forward(x)
print(y)
z=ac.backward(err_x)
print(z)
'''


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        # fix the max value overflow
        self.input = x - torch.max(x, dim=1, keepdim=True)[0]

        # Compute the exponent of the input tensor
        exp = torch.exp(self.input)

        # Compute the sum of the exponent over the specified dimension
        sum = torch.sum(exp, dim=1, keepdim=True)

        # Divide the exponent by the sum to compute the softmax
        return exp / sum

    def backward(self, grad_output):
        # Compute the exponent of the input tensor
        exp = torch.exp(self.input)

        # Compute the sum of the exponent over the specified dimension
        sum = torch.sum(exp, dim=1, keepdim=True)

        # Compute the Jacobian matrix of the softmax function
        jacobian = exp / (sum ** 2)

        # Compute the gradient of the softmax with respect to the input
        grad_input = jacobian * grad_output

        return grad_input


'''
somfmax_layer = Softmax()
x = torch.randn([2, 10])
erro_dev = torch.randn([2, 10])
pred = somfmax_layer.forward(x)
print(pred.shape)
grad_input = somfmax_layer.backward(erro_dev)
print(grad_input.shape)
'''


class MSELoss:
    def __init__(self):
        pass

    def forward(self, input, target):
        # Calculate the mean squared error between the input and target tensors
        self.input = input  # Save the input tensor for use in the backward pass
        self.target = target
        return ((self.input - self.target) ** 2).mean(dim=-1).unsqueeze(-1)

    def backward(self):
        # Calculate the gradient of the loss with respect to the input tensor
        return ((self.input - self.target) * 2).mean(dim=-1).unsqueeze(-1)


'''
pred = torch.randn([2, 10])
target = torch.randn([2, 10])
critation = MSELoss()
loss = critation.forward(pred, target)
print(loss.sum(), loss.shape)
erro_dev = critation.backward()
print(erro_dev.shape)
'''


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = torch.randn(out_channels)

        self.params = {"weight": self.weight, "bias": self.bias}
        self.params_grad = {"weight": None, "bias": None}

    def forward(self, input):
        self.input = input
        n, c, h, w = self.input.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        x_padded = torch.zeros((n, c, h + 2 * self.padding, w + 2 * self.padding))
        x_padded[:, :, self.padding: self.padding + h, self.padding: self.padding + w] = self.input

        out = torch.zeros((n, self.out_channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                x_window = x_padded[:, :, i * self.stride: i * self.stride + self.kernel_size,
                           j * self.stride: j * self.stride + self.kernel_size]
                for k in range(self.out_channels):
                    out[:, k, i, j] = torch.sum(x_window * self.weight[k, :, :, :], dim=(1, 2, 3))

        out += self.bias[:, None, None]

        return out

    def backward(self, grad_output):
        n, c, h, w = self.input.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        x_padded = torch.zeros((n, c, h + 2 * self.padding, w + 2 * self.padding))
        x_padded[:, :, self.padding: self.padding + h, self.padding: self.padding + w] = self.input

        grad_input = torch.zeros_like(x_padded)
        grad_weight = torch.zeros_like(self.weight)
        grad_bias = torch.zeros_like(self.bias)

        for i in range(out_h):
            for j in range(out_w):
                x_window = x_padded[:, :, i * self.stride: i * self.stride + self.kernel_size,
                           j * self.stride: j * self.stride + self.kernel_size]
                for k in range(self.out_channels):
                    # grad for previous layer
                    grad_input[:, :, i * self.stride: i * self.stride + self.kernel_size,
                    j * self.stride: j * self.stride + self.kernel_size] += \
                        grad_output[:, k, i, j][:, None, None, None] * self.weight[k, :, :, :]

                    # grad for kernal (aggrate the batch)
                    for b in range(grad_output.shape[0]):
                        grad_weight[k, :, :, :] += (grad_output[:, k, i, j][:, None, None, None] * x_window)[b]
        # cut region
        if self.padding == 0:
            pass
        else:
            grad_input = grad_input[:, :, self.padding: -self.padding, self.padding: -self.padding]
        grad_bias = torch.sum(grad_output, dim=(0, 2, 3))

        # update grad for pamrameter
        self.params_grad = {"weight": grad_weight, "bias": grad_bias}

        return grad_input


# Define the SGD optimizer class
class SGD:
    def __init__(self, model, lr):
        # Save the list of parameters and the learning rate as attributes of the SGD instance
        self.layers = []
        for depth_idx, layer in enumerate(model.layers):
            try:
                layer_params = layer.params  # params.dict
                layer_params_grad = layer.params_grad
            except:
                pass  # no parames
            else:
                self.layers.append(depth_idx)

        self.lr = lr

        self.model = model

    def step(self):
        # Iterate over the list of parameters
        for depth_idx in self.layers:
            param = self.model.layers[depth_idx].params
            # Update the parameter tensor by subtracting the product of the learning rate
            # and the gradient of the loss with respect to the parameter tensor from the parameter tensor
            for param_key in param:
                param[param_key] -= self.lr * self.model.layers[depth_idx].params_grad[param_key]

                # clean grad
                self.model.layers[depth_idx].params_grad[param_key] = None


class Adam:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999)):
        self.model = model
        self.layers = []

        self.m = {}
        self.v = {}

        for depth_idx, layer in enumerate(model.layers):
            try:
                layer_params = layer.params  # params.dict
                layer_params_grad = layer.params_grad
            except:
                pass  # no parames
            else:
                self.layers.append(depth_idx)
                self.m[depth_idx] = {}
                self.v[depth_idx] = {}

                for param_key in self.model.layers[depth_idx].params:
                    self.m[depth_idx][param_key] = torch.zeros_like(self.model.layers[depth_idx].params[param_key])
                    self.v[depth_idx][param_key] = torch.zeros_like(self.model.layers[depth_idx].params[param_key])

        self.lr = lr
        self.betas = betas

    def step(self):
        # Iterate over the list of parameters
        for depth_idx in self.layers:
            param = self.model.layers[depth_idx].params
            # Update the parameter tensor by subtracting the product of the learning rate
            # and the gradient of the loss with respect to the parameter tensor from the parameter tensor
            for param_key in param:
                self.m[depth_idx][param_key] = self.betas[0] * self.m[depth_idx][param_key] + \
                                               (1 - self.betas[0]) * self.model.layers[depth_idx].params_grad[param_key]
                self.v[depth_idx][param_key] = self.betas[1] * self.v[depth_idx][param_key] + (1 - self.betas[1]) * \
                                               self.model.layers[depth_idx].params_grad[param_key] ** 2
                self.model.layers[depth_idx].params[param_key] -= self.lr * self.m[depth_idx][param_key] \
                                                                  / (self.v[depth_idx][param_key].sqrt() + 1e-8)

                # clean grad
                self.model.layers[depth_idx].params_grad[param_key] = None


class Network:
    def __init__(self):
        self.layers = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def forward(self, x):
        # forward a batch

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, erro_dev):

        for layer in reversed(self.layers):
            erro_dev = layer.backward(erro_dev)

        return erro_dev


'''
X, Y = get_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# MLP
net = Network()
net.add(FC(10, 24))
net.add(ReLU())
net.add(FC(24, 24))
net.add(ReLU())
net.add(FC(24, 24))
net.add(ReLU())
net.add(FC(24, 1))
net.add(ReLU())

optimizer = SGD(net, lr=0.0001)
pred = net.forward(X_train[0])

critation = MSELoss()
loss = critation.forward(pred, Y_train[0])
print(loss.sum())
erro_dev = critation.backward()

net.backward(erro_dev)
optimizer.step()

pred = net.forward(X_train[0])
loss = critation.forward(pred, Y_train[0])
print(loss.sum())
'''

'''
# CNN
x = torch.randn(2, 3, 224, 224)

net = Network()
net.add(Conv2D(in_channels=3, out_channels=16, kernel_size=16, stride=16, padding=16))
net.add(Reshape((16, 16, 16), (4096)))
net.add(FC(4096, 24))
net.add(ReLU())
net.add(FC(24, 1))
net.add(ReLU())

critation = MSELoss()
optimizer = Adam(net, lr=0.0001)

pred = net.forward(x)
print(pred.shape)
target = torch.randn(pred.shape)

loss = critation.forward(pred, target)
print(loss.sum(),loss.shape)

grad_output = critation.backward()
print(grad_output.shape)
net.backward(grad_output)

optimizer.step()

output = net.forward(x)
loss = critation.forward(pred, target)
print(loss.sum())
'''


def train(model, train_loader, test_loader, critation, optimizer, epochs=20, flattern=True):
    Train_loss_rec = []
    Val_loss_rec = []

    for epoch in range(epochs):
        # train
        accum_loss = 0.0
        for input_batch_img, batch_label in tqdm(train_loader):
            input, label = minist_preprocess(input_batch_img, batch_label, flattern=flattern)

            # forward
            pred = model.forward(input)

            loss = critation.forward(pred, label)
            accum_loss += loss.sum()

            # back propergation
            de_loss = critation.backward()
            model.backward(de_loss)

            optimizer.step()

        Train_loss_rec.append(float((accum_loss / len(train_loader))))
        print('In epoch ', epoch + 1, 'Train loss', Train_loss_rec[epoch])

        # val
        accum_loss = 0.0
        for input_batch_img, batch_label in test_loader:
            input, label = minist_preprocess(input_batch_img, batch_label, flattern=flattern)
            pred = model.forward(input)
            loss = critation.forward(label, pred)
            accum_loss += loss.sum()
        Val_loss_rec.append(float((accum_loss / len(test_dataset))))
        print('In epoch ', epoch + 1, 'Val loss', Val_loss_rec[epoch])
        print()

    return model, Train_loss_rec, Val_loss_rec


if __name__ == '__main__':
    # Set the batch size and number of epochs
    batch_size = 8
    num_epochs = 10

    # Create a transform that normalizes the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST dataset using the specified transform
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create a data loader for the training and test datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    '''
    # MLP
    model = Network()
    model.add(FC(784, 392))
    model.add(ReLU())
    model.add(FC(392, 196))
    model.add(ReLU())
    model.add(FC(196, 49))
    model.add(ReLU())
    model.add(FC(49, 10))
    model.add(Softmax())
    '''

    # CNN
    model = Network()
    model.add(Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1))
    model.add(ReLU())
    model.add(Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
    model.add(ReLU())
    model.add(Conv2D(in_channels=128, out_channels=10, kernel_size=1, stride=1, padding=0))
    model.add(ReLU())
    model.add(Reshape((10, 7, 7), (490)))
    model.add(FC(490, 200))
    model.add(FC(200, 50))
    model.add(FC(50, 10))
    model.add(Softmax())

    # loss
    critation = MSELoss()

    optimizer = Adam(model, lr=0.0001)

    # train
    model, Train_loss_rec, Val_loss_rec = train(model, train_loader, test_loader, critation, optimizer,
                                                epochs=20, flattern=False)

    draw(Train_loss_rec[-10:], Val_loss_rec[-10:])
