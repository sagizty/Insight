import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def draw(train_res, val_res):
    plt.plot(train_res, 'r',label='Training')
    plt.plot(val_res, 'b',label='Testing')
    plt.ylabel('loss (MSE)')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.title('MLP fitting curve')
    plt.show()


def get_data(size=100, dim=10):
    dataset = torch.normal(mean=0, std=1, size=(size,))

    Y = []
    X = []
    for i in range(size):
        index = torch.randint(dataset.shape[0], size=(dim,))
        Y.append(dataset[index].mean())
        X.append(dataset[index])

    return X, Y


# with nn.linear
class MLP_model(nn.Module):
    def __init__(self, in_features=10, hidden_features=24, out_features=1, depth=2, act_layer=nn.ReLU):
        super().__init__()
        self.depth = depth

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.act_layer = act_layer()

        self.fc_first = nn.Linear(in_features, hidden_features)

        for k in range(depth):
            exec('self.fc' + str(k) + '=nn.Linear(' + str(hidden_features) + ',' + str(hidden_features) + ')')

        self.fc_last = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc_first(x)

        for k in range(self.depth):
            hidden_layer = eval('self.fc' + str(k))
            x = hidden_layer(x)
            x = self.act_layer(x)

        x = self.fc_last(x)

        return x


def hook_forward_function(module, module_input, module_output):
    print('forward module_input', module_input)
    print('forward module_output', module_output)


def hook_backward_function(module, module_input_grad, module_output_gard):
    print('backward module_input_grad', module_input_grad)
    print('backward module_output_gard', module_output_gard)


def train(model, X_train, Y_train, X_test, Y_test, critation, optimizer, epochs=20):
    Train_loss_rec = []
    Val_loss_rec = []

    for epoch in range(epochs):
        # train
        accum_loss = 0
        for i in range(len(X_train)):
            model.zero_grad()

            # track the input with autograd Variable()
            input, label = Variable(X_train[i]), Variable(Y_train[i])
            pred = model(input)

            loss = critation(pred, label)
            accum_loss += loss

            # print(input.grad)
            # print(pred.grad)

            # back propergation
            loss.backward()

            # print(input.grad)
            # print(pred.grad)
            optimizer.step()

            # print(input.grad)
            # print(pred.grad)

        Train_loss_rec.append(accum_loss.item() / len(X_train))

        # val
        accum_loss = 0
        for i in range(len(X_test)):
            input, label = Variable(X_test[i]), Variable(Y_test[i])
            pred = model(input)
            loss = critation(pred, label)
            accum_loss += loss
        Val_loss_rec.append(accum_loss.item() / len(X_test))

    return model, Train_loss_rec, Val_loss_rec


X, Y = get_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

MLP = MLP_model(in_features=10, hidden_features=24, out_features=1, depth=2, act_layer=nn.ReLU)

# track and print the grad (forward and backward), take fc_first as the example
# h1 = MLP.fc_first.register_forward_hook(hook_forward_function)
# h2 = MLP.fc_first.register_backward_hook(hook_backward_function)

critation = nn.MSELoss()
optimizer = torch.optim.SGD(MLP.parameters(), lr=0.01)

model, Train_loss_rec, Val_loss_rec = train(MLP, X_train, Y_train, X_test, Y_test, critation, optimizer, epochs=50)

# remove hook
# h1.remove()
# h2.remove()

draw(Train_loss_rec, Val_loss_rec)

model, Train_loss_rec, Val_loss_rec = train(model, X_train, Y_train, X_test, Y_test, critation, optimizer, epochs=5)
# now no triggering of the hook print

