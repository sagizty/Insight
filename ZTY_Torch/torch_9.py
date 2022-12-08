import torch
import torch.nn as nn
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
    dataset = torch.normal(mean=0, std=1, size=(size,))

    X = []
    Y = []
    for i in range(size):
        index = torch.randint(dataset.shape[0], size=(dim,))

        x = dataset[index]
        y = x.mean().unsqueeze(0)

        X.append(x.unsqueeze(0))
        Y.append(y.unsqueeze(0))

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


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


def ReLU(x):
    return torch.clamp(x, min=0)


def ReLU_deriv(Z):
    # set > 0 to 1, rest to 0
    return Z > 0


class ActivationLayer(Layer):
    def __init__(self, activation=ReLU, activation_prime=ReLU_deriv):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data  # [B,Dim]
        x = self.activation(self.input)
        return x

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate=None):
        # [B,Dim] -> [B,Dim]
        return self.activation_prime(self.input) * output_error


'''
x = torch.randn([2,10])
err_x=torch.randn([2,10])
ac=ActivationLayer()
y=ac.forward_propagation(x)
print(y)
z=ac.backward_propagation(err_x)
print(z)
'''


class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.input_size = input_size  # Dim
        self.output_size = output_size  # out_Dim

        self.weights = torch.normal(mean=0, std=1, size=(input_size, output_size))
        self.bias = torch.normal(mean=0, std=1, size=(1, output_size))

    # returns output for a given input
    def forward_propagation(self, input_data):
        B, dim = input_data.shape
        self.input = input_data
        # [B,Dim] (*[Dim,out_Dim]) -> [B,out_Dim] (+[B,out_Dim]) -> [B,out_Dim]
        x = torch.mm(self.input, self.weights) + self.bias.expand(B, -1)
        return x

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        B, out_Dim = output_error.shape
        # [B,out_Dim] -> [B,Dim] for previous block
        back_error = torch.mm(output_error, self.weights.T)
        # [B,Dim] -> [Dim,B] * [B,out_Dim] -> [Dim,out_Dim]
        weights_error = torch.mm(self.input.T, output_error)
        # [B,out_Dim] -> [out_Dim] -> [1,out_Dim]
        dBias = torch.sum(output_error, 0).unsqueeze(0)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * dBias
        return back_error


'''
# test forward and backward

mod = FCLayer(10,24)
x = torch.randn([2,10])
err_x = torch.randn([2,24])
y = mod.forward_propagation(x)
mod.backward_propagation(err_x,0.001)
print(y)
y = mod.forward_propagation(x)
print(y)
'''


# loss function and its derivative
def mse(target, pred):
    # MSE loss
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1).unsqueeze(-1)

    return loss


def mse_deriv(target, pred):
    de_loss = 2 * (pred - target)
    de_loss = de_loss.mean(dim=-1).unsqueeze(-1)
    return de_loss


class loss:
    def __init__(self):
        self.target = None
        self.pred = None

    # computes the output Y of a layer for a given input X
    def forward_measure(self, target, pred):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_measure(self):
        raise NotImplementedError


class MSE(loss):
    def __init__(self, loss_measure=mse, loss_prime=mse_deriv):
        self.loss_measure = loss_measure
        self.loss_prime = loss_prime

    # returns the activated input
    def forward_measure(self, target, pred):
        loss = self.loss_measure(target, pred)
        self.target = target
        self.pred = pred
        return loss

    # Returns de_loss=dE/dX for a given output_error=dE/dY.
    def backward_measure(self):
        de_loss = self.loss_prime(self.target, self.pred)
        return de_loss


'''
X, Y = get_data()
critation = MSE()
loss = critation.forward_measure(Y[0], pred)
print(loss.sum(),loss.shape)
de_loss = critation.backward_measure()
print(de_loss.shape)
'''


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def forward(self, x):
        # forward a batch

        for layer in self.layers:
            x = layer.forward_propagation(x)

        return x

    def backward(self, de_loss, learning_rate=0.0001):

        for layer in reversed(self.layers):
            de_loss = layer.backward_propagation(de_loss, learning_rate)


'''
X, Y = get_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# netork
net = Network()
net.add(FCLayer(10, 24))
net.add(ActivationLayer(ReLU, ReLU_deriv))
net.add(FCLayer(24, 24))
net.add(ActivationLayer(ReLU, ReLU_deriv))
net.add(FCLayer(24, 1))
net.add(ActivationLayer(ReLU, ReLU_deriv))

pred = net.forward(X_train[0])
critation = MSE()
loss = critation.forward_measure(Y_train[0], pred)
print(loss.sum())
de_loss = critation.backward_measure()

net.backward(de_loss, learning_rate=0.0001)

pred = net.forward(X_train[0])
loss = mse(Y_train[0], pred)
print(loss.sum())
'''


def train(model, X_train, Y_train, X_test, Y_test, critation, epochs=20, lr=0.00001):
    Train_loss_rec = []
    Val_loss_rec = []

    for epoch in range(epochs):
        # train
        accum_loss = 0.0
        for i in range(len(X_train)):
            input, label = X_train[i], Y_train[i]

            # forward
            pred = model.forward(input)

            loss = critation.forward_measure(label, pred)
            accum_loss += loss.sum()

            # back propergation
            de_loss = critation.backward_measure()

            model.backward(de_loss, learning_rate=lr)

        Train_loss_rec.append(float((accum_loss / len(X_train))))

        # val
        accum_loss = 0.0
        for i in range(len(X_test)):
            input, label = X_test[i], Y_test[i]
            pred = model.forward(input)
            loss = critation.forward_measure(label, pred)
            accum_loss += loss.sum()
        Val_loss_rec.append(float((accum_loss / len(X_test))))

    return model, Train_loss_rec, Val_loss_rec


X, Y = get_data(size=1000, dim=10, batch_size=8)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# netork
model = Network()
model.add(FCLayer(10, 24))
model.add(ActivationLayer(ReLU, ReLU_deriv))
model.add(FCLayer(24, 24))
model.add(ActivationLayer(ReLU, ReLU_deriv))
model.add(FCLayer(24, 24))
model.add(ActivationLayer(ReLU, ReLU_deriv))
model.add(FCLayer(24, 1))
model.add(ActivationLayer(ReLU, ReLU_deriv))

# loss
critation = MSE()

# train
model, Train_loss_rec, Val_loss_rec = train(model, X_train, Y_train, X_test, Y_test, critation,
                                            epochs=20, lr=0.0000001)

draw(Train_loss_rec, Val_loss_rec)
