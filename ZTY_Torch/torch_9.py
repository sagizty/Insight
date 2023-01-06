"""
use torch tensor to build a CNN network
with only tensor
without autograd and everything

"""

import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def draw(train_res, val_res):
    plt.plot(train_res, 'r', label='Training')
    plt.plot(val_res, 'b', label='Testing')
    plt.ylabel('loss (MSE)')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.title('MLP fitting curve')
    plt.show()


def get_data(size=1000, dim=10, batch_size=8):
    # dataset
    dataset = torch.normal(mean=0, std=1, size=(size,))

    X = []
    Y = []
    for i in range(size):
        index = torch.randint(dataset.shape[0], size=(dim,))

        x = dataset[index]
        y = x.mean().unsqueeze(0)

        X.append(x.unsqueeze(0))
        Y.append(y.unsqueeze(0))

    # dataloader
    Batch_X = []
    Batch_Y = []

    batch_num = size // batch_size

    for batch in range(batch_num):
        x_batch = X[batch * batch_size:(batch + 1) * batch_size]
        y_batch = Y[batch * batch_size:(batch + 1) * batch_size]
        x = torch.stack(x_batch).squeeze()
        y = torch.stack(y_batch).squeeze()
        if len(y.shape) < len(x.shape):
            y = y.unsqueeze(-1)
        Batch_X.append(x)
        Batch_Y.append(y)

    return Batch_X, Batch_Y


class FC:
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.randn(output_size)

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
mod.backward(err_x)
mod.update(0.001)
print(y)
y = mod.forward(x)
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


'''
X, Y = get_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# netork
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


def train(model, X_train, Y_train, X_test, Y_test, critation, optimizer, epochs=20):
    Train_loss_rec = []
    Val_loss_rec = []

    for epoch in range(epochs):
        # train
        accum_loss = 0.0
        for i in range(len(X_train)):
            input, label = X_train[i], Y_train[i]

            # forward
            pred = model.forward(input)

            loss = critation.forward(pred,label)
            accum_loss += loss.sum()

            # back propergation
            de_loss = critation.backward()
            model.backward(de_loss)

            optimizer.step()

        Train_loss_rec.append(float((accum_loss / len(X_train))))

        # val
        accum_loss = 0.0
        for i in range(len(X_test)):
            input, label = X_test[i], Y_test[i]
            pred = model.forward(input)
            loss = critation.forward(label, pred)
            accum_loss += loss.sum()
        Val_loss_rec.append(float((accum_loss / len(X_test))))

    return model, Train_loss_rec, Val_loss_rec


if __name__ == '__main__':
    X, Y = get_data(size=1000, dim=10, batch_size=8)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # netork
    model = Network()
    model.add(FC(10, 24))
    model.add(ReLU())
    model.add(FC(24, 24))
    model.add(ReLU())
    model.add(FC(24, 24))
    model.add(ReLU())
    model.add(FC(24, 1))
    model.add(ReLU())

    # loss
    critation = MSELoss()

    optimizer = SGD(model, lr=0.00001)

    # train
    model, Train_loss_rec, Val_loss_rec = train(model, X_train, Y_train, X_test, Y_test, critation, optimizer,
                                                epochs=20)

    draw(Train_loss_rec, Val_loss_rec)
