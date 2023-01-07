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


def draw_curve(train_res, val_res, pic_name='Generator fitting curve', draw_path='./imaging_results'):
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
            if inputs.shape[0] >8:
                num_images = 8

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


# Generator
class AutoEncoder(nn.Module):
    def __init__(self, latent_space_dim=20):
        super(AutoEncoder, self).__init__()

        # encoder to encode latent fatures
        # [b, 784] -> [b, hidden_1] -> [b, hidden_2] -> [b, hidden_3]
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_space_dim)
        )

        # decoder to decode latent fatures
        # [b, hidden_3] -> [b, hidden_2] -> [b, hidden_1] -> [b, 784]
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # encoder
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        latent = self.encoder_lin(x)

        # decoder
        x = self.decoder_lin(latent)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x, latent


# Discriminator
class Discriminator_model(nn.Module):

    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=0)
        #self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

        # a fake residual connection
        #self.res_conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)

        #self.relu = nn.ReLU()

        self.MLP = nn.Sequential(
            nn.Linear(784, 98),
            nn.ReLU(),
            nn.Linear(98, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # process a bag(as a batch)
        # [B, 1, 28, 28]

        #x_res = self.res_conv(x)
        #x = self.relu(self.conv1(x))
        #x = self.relu(self.conv2(x))
        #x = self.relu(self.conv3(x))

        #x += x_res  # a fake residual connection
        B, C, H, W = x.shape
        x = x.view(-1, C * H * W)

        x = self.MLP(x)

        return x


def train(Generator, Discriminator, train_loader, test_loader, criterion, rec_criterion, optimizerG, optimizerD,
          epochs=20,
          draw_path='./imaging_results', device="cpu"):
    if not os.path.exists(draw_path):
        os.makedirs(draw_path)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    Train_G_losses_rec = []
    Train_rec_losses_rec = []
    Train_D_losses_rec = []
    Val_G_losses_rec = []
    Val_rec_losses_rec = []
    Val_D_losses_rec = []
    epoch_dfs = []
    # ittr over epoches
    for epoch in range(epochs):
        # train
        Generator.train()
        Discriminator.train()
        accum_loss = 0.0

        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader)):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            D_balance = 1.0
            Discriminator.zero_grad()
            # inputs: [b, 1, 28, 28]
            inputs = inputs.to(device)
            identity_label = torch.full((inputs.shape[0],), real_label, dtype=torch.float, device=device)

            # (a) Train D with original input
            identity_output = Discriminator(inputs).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(identity_output, identity_label) * D_balance
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = identity_output.mean().item()

            # (b) Train Discriminator with blur_noise batch
            blur_noise = torch.randn_like(inputs, device=device)
            blur_inputs = inputs + blur_noise*0.1  # Generate batch of blur inputs

            # Generate blur_inputs image batch
            identity_label.fill_(fake_label)
            # Classify all blur batch with D
            identity_output = Discriminator(blur_inputs).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_blur = criterion(identity_output, identity_label) * D_balance
            # Calculate the gradients for this batch
            errD_blur.backward()
            D_blur = identity_output.mean().item()

            # (c) Train Discriminator with reconstracted batch
            reconstructed, latent = Generator(inputs)
            identity_label.fill_(fake_label)
            # Classify all fake batch with D (detach don't torch G)
            identity_output = Discriminator(reconstructed.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(identity_output, identity_label) * D_balance
            D_G_z1 = identity_output.mean().item()
            errD_fake.backward()

            # Add the gradients from the all-real and all-fake batches
            errD = (errD_real + errD_blur) / 2 + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z))) + minimize rec_loss
            ############################
            Generator.zero_grad()
            identity_label.fill_(real_label)  # fake labels are real for generator cost
            # Since we already updated D, perform another forward pass of all-fake batch through D
            identity_output = Discriminator(reconstructed).view(-1)
            # Calculate D -> G's loss based on this output
            errG = criterion(identity_output, identity_label) * D_balance
            D_G_z2 = identity_output.mean().item()
            # rec loss
            rec_loss = rec_criterion(reconstructed, inputs)
            accum_loss += rec_loss.item()

            # Calculate gradients for G
            G_loss = errG + rec_loss
            G_loss.backward()

            # Update G
            optimizerG.step()

            if batch_idx % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t reconstraction loss: %.4f'
                      '\nD(x): %.4f\tD_blur: %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, epochs, batch_idx, len(train_loader), errD.item(), errG.item(),
                         rec_loss, D_x, D_blur, D_G_z1, D_G_z2))

        Train_loss = accum_loss / (len(train_loader) * inputs.shape[0])
        print("Epoch: ", epoch + 1, "Train Loss: ", Train_loss)
        # Save Losses for plotting later
        Train_G_losses_rec.append(errG.item())
        Train_rec_losses_rec.append(Train_loss)
        Train_D_losses_rec.append(errD.item())
        # visualize the last one of test to check
        visualize_check(inputs, pic_name='Train ori at epoch ' + str(epoch + 1))
        visualize_check(blur_inputs, pic_name='Train blur at epoch ' + str(epoch + 1))
        visualize_check(reconstructed, pic_name='Train rec at epoch ' + str(epoch + 1))

        # Test
        Generator.eval()
        Discriminator.eval()
        feature_latent = None
        feature_labels = None
        accum_loss = 0.0
        for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                inputs = inputs.to(device)
                reconstructed, latent = Generator(inputs)
                loss = rec_criterion(reconstructed, inputs)
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
        Val_rec_losses_rec.append(Val_loss)

        # visualize the last one of test to check
        visualize_check(inputs, pic_name='Test ori at epoch ' + str(epoch + 1))
        visualize_check(reconstructed, pic_name='Test rec at epoch ' + str(epoch + 1))

    return Generator, Train_rec_losses_rec, Val_rec_losses_rec, epoch_dfs


if __name__ == '__main__':
    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Set the batch size and number of epochs
    latent_space_dim = 128
    batch_size = 50
    num_epochs = 100
    lr = 1e-4  #-6 ->-5

    # Create a transform that normalizes the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST dataset using the specified transform
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create a data loader for the training and test datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    Generator = AutoEncoder(latent_space_dim=latent_space_dim)
    Discriminator = Discriminator_model()
    # put to device
    Generator.to(device)
    Discriminator.to(device)

    optimizerG = torch.optim.Adam(Generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(Discriminator.parameters(), lr=lr)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    rec_criterion = L1_norm_reconstruction_loss()

    Generator, Train_loss_rec, Val_loss_rec, epoch_dfs = train(Generator, Discriminator, train_loader, test_loader,
                                                               criterion, rec_criterion, optimizerG, optimizerD,
                                                               epochs=num_epochs, device=device)
    # draw_curve(Train_loss_rec, Val_loss_rec)
