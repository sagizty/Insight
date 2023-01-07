import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')


class L1_norm_reconstruction_loss(nn.L1Loss):
    """
    element-wise l1 loss for reconstruction
    """

    def __init__(self):
        super(L1_norm_reconstruction_loss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(torch.flatten(input, start_dim=1), torch.flatten(target, start_dim=1),
                         reduction=self.reduction)


def draw_curve(train_res, val_res, pic_name='AutoEncoder fitting curve', draw_path='./imaging_results'):
    plt.plot(range(1, len(train_res) + 1), train_res, "b", label='Train')
    plt.plot(range(1, len(val_res) + 1), val_res, "r", label='Val')
    plt.ylabel('loss (MSE)')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.title(pic_name)
    picpath = os.path.join(draw_path, pic_name + '.jpg')
    if not os.path.exists(draw_path):
        os.makedirs(draw_path)
    plt.savefig(picpath)
    plt.show()


def imshow(inp, title=None):  # Imshow for Tensor
    """Imshow for Tensor."""
    inp = torch.cat((inp, inp, inp), 0)
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_check(inputs, num_images=-1, pic_name='test', draw_path='./imaging_results', writer=None):  # visual check
    """
    check num_images of images and visual them
    output a pic with 3 column and rows of num_images//3

    :param inputs: inputs of data
    :param num_images: how many image u want to check, should SMALLER THAN the batchsize
    :param pic_name: name of the output pic
    :param draw_path: path folder for output pic
    :param writer: attach the pic to the tensorboard backend

    :return:  None

    """

    images_so_far = 0
    plt.figure()

    with torch.no_grad():

        if num_images == -1:  # auto detect a batch
            num_images = int(inputs.shape[0])

        if num_images % 5 == 0:
            line_imgs_num = 5
        elif num_images % 4 == 0:
            line_imgs_num = 4
        elif num_images % 3 == 0:
            line_imgs_num = 3
        elif num_images % 2 == 0:
            line_imgs_num = 2
        else:
            line_imgs_num = int(num_images)

        rows_imgs_num = int(num_images // line_imgs_num)
        num_images = line_imgs_num * rows_imgs_num

        for j in range(num_images):  # each batch input idx: j

            images_so_far += 1

            ax = plt.subplot(rows_imgs_num, line_imgs_num, images_so_far)

            ax.axis('off')
            # ax.set_title()
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                picpath = os.path.join(draw_path, pic_name + '.jpg')
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                '''
                myfig = plt.gcf()  # get current image
                myfig.savefig(picpath, dpi=1000)
                '''
                plt.savefig(picpath)
                plt.show()

                if writer is not None:  # attach the pic to the tensorboard backend if avilable
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                plt.cla()
                plt.close("all")
                return


class AutoEncoder(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=64, hidden_3=20):
        super(AutoEncoder, self).__init__()

        # encoder to encode latent fatures
        # [b, 784] -> [b, hidden_1] -> [b, hidden_2] -> [b, hidden_3]
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU()
        )

        # decoder to decode latent fatures
        # [b, hidden_3] -> [b, hidden_2] -> [b, hidden_1] -> [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_3, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [b, 1, 28, 28]
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size, -1)
        # encoder
        latent = self.encoder(x)
        # decoder
        x = self.decoder(latent)
        # reshape to img
        x = x.view(batch_size, 1, 28, 28)

        return x, latent


def train(model, train_loader, test_loader, criterion, optimizer, epochs=20,
          draw_path='./imaging_results', device="cpu"):
    Train_loss_rec = []
    Val_loss_rec = []
    epoch_dfs = []
    # ittr over epoches
    for epoch in range(epochs):
        # train
        model.train()
        accum_loss = 0.0
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader)):
            model.zero_grad()
            # inputs: [b, 1, 28, 28]
            inputs = inputs.to(device)

            reconstructed, latent = model(inputs)
            loss = criterion(reconstructed, inputs)
            accum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Train_loss = accum_loss / (len(train_loader) * inputs.shape[0])
        print("Epoch: ", epoch + 1, "Train Loss: ", Train_loss)
        Train_loss_rec.append(Train_loss)

        # Test
        model.eval()
        feature_latent = None
        feature_labels = None
        accum_loss = 0.0
        for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                inputs = inputs.to(device)
                reconstructed, latent = model(inputs)
                loss = criterion(reconstructed, inputs)
                accum_loss += loss.item()

                # build tensor for log df
                labels = labels.unsqueeze(-1)  # [B] -> [B, 1]
                feature_labels = torch.cat((feature_labels, labels), dim=0) if feature_latent is not None else labels
                feature_latent = torch.cat((feature_latent, latent), dim=0) if feature_latent is not None else latent

        # log df (only for test)
        df_labels = pd.DataFrame(feature_labels.cpu().numpy())
        df_labels.columns = ['label']
        df_latent = pd.DataFrame(feature_latent.cpu().numpy())
        epoch_dfs.append({'labels': df_labels, 'latent': df_latent})
        df = pd.concat([df_labels, df_latent], axis=1)
        csv_path = os.path.join(draw_path, 'latent_' + str(latent.shape[-1]) + '_Epoch_' + str(epoch + 1) + '.csv')
        df.to_csv(csv_path, index=False)

        Val_loss = accum_loss / (len(test_loader) * inputs.shape[0])
        print("Epoch: ", epoch + 1, "Test Loss: ", Val_loss)
        Val_loss_rec.append(Val_loss)

        # visualize the last one of test to check
        visualize_check(inputs, pic_name='ori at epoch ' + str(epoch + 1))
        visualize_check(reconstructed, pic_name='rec at epoch ' + str(epoch + 1))

    return model, Train_loss_rec, Val_loss_rec, epoch_dfs


if __name__ == '__main__':
    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Set the batch size and number of epochs
    hidden_1 = 256
    hidden_2 = 32
    hidden_3 = 2
    batch_size = 8
    num_epochs = 10
    lr = 1e-3
    criterion = nn.MSELoss()  # L1_norm_reconstruction_loss + lr=1e-4 + 50 epochs perform worse
    # than nn.MSELoss() + lr=1e-3 + 5 epochs

    # Create a transform that normalizes the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST dataset using the specified transform
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create a data loader for the training and test datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoEncoder(hidden_1=hidden_1, hidden_2=hidden_2, hidden_3=hidden_3)
    model.to(device)  # put to device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, Train_loss_rec, Val_loss_rec, epoch_dfs = train(model, train_loader, test_loader, criterion,
                                                           optimizer, epochs=num_epochs, device=device)
    draw_curve(Train_loss_rec, Val_loss_rec)
