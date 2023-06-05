"""
Second course is about dataframe
the dataset dataloader and data transforms

https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))

cols, rows = 3, 3

for i in range(0, cols * rows):

    sample_idx = torch.randint(len(training_data), size=(1,)).item()

    img, label = training_data[sample_idx]

    figure.add_subplot(rows, cols, i+1)

    plt.title(labels_map[label])

    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()



from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")