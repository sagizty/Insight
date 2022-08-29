"""
A project sample
Script ver: 08/08/2022
Author Zhang Tianyi
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import timm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from model.ViT import VisionTransformer


def setup_seed(seed):  # setting up the random seed for reproduction
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def imshow(inp, title=None):  # Imshow for Tensor input
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    '''
    if required: Alter the transform 
    because transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(confidence, preds, inputs, labels, class_names, num_images=4, picpath='../imaging_results', draw_idx=0):

    if not os.path.exists(picpath):
        os.makedirs(picpath)

    images_so_far = 0
    fig = plt.figure()

    for j in range(inputs.size()[0]):
        images_so_far += 1
        ax = plt.subplot(num_images // 2, 2, images_so_far)
        ax.axis('off')
        ax.set_title(f'Label: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}\nConf: {confidence[j]}')
        imshow(inputs.cpu().data[j])
        plt.pause(0.001)

        if images_so_far == num_images:
            picpath = os.path.join(picpath, str(draw_idx)+'_output.jpg')
            plt.savefig(picpath, dpi=1000)


def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch_idx = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # refresh best epoch : deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_epoch_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    print(f'Best Epoch: ',best_epoch_idx )
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, dataloader, criterion, class_names, dataset_size, check_minibatch=10, device='cuda'):
    model.eval()
    print('Test')
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0
    check_idx = 0
    draw_idx = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        check_idx += 1

        # forward
        outputs = model(inputs)
        confidence, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        if check_idx % check_minibatch == 0:
            draw_idx += 1
            visualize_model(confidence, preds, inputs, labels, class_names, num_images=4, draw_idx=draw_idx)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


# Step 0 : enviroment preparing
setup_seed(42)
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Finished Step 0 : enviroment preparing')


# Step 1 : data preparing
data_dir = '../data/warwick_CLS'
img_size = 224
num_classes = 2

data_transforms = {
    # Data augmentation and normalization for training
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Just normalization for validation
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['val'])

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
test_dataset_size = len(test_dataset)

class_names = [d.name for d in os.scandir(os.path.join(data_dir, 'train')) if d.is_dir()]

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

print('Finished Step 1 : data preparing')


# Step 2 : build a model (Transfer learning)
model = VisionTransformer(img_size=img_size, num_classes=num_classes, pretrained=True)
'''
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
'''
model.to(device)

print('Finished Step 2 : build a model (Transfer learning)')


# Step 3 : set optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print('Finished Step 3 : set optimizer and loss')


# Step 4: start training loop
model = train_model(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes,
                    num_epochs=10, device=device)
print('Finished Step 4: start training loop')

# Step 5: test model
test_model(model, test_dataloader, criterion, class_names, test_dataset_size, check_minibatch=10, device=device)
print('Finished Step 5: test model')
