import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from torchvision import datasets, transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def split_data(dataset):
    test_pct = 0.2
    val_pct = 0.1

    test_size = int(len(dataset) * test_pct)
    dataset_size = len(dataset) - test_size
    val_size = int(dataset_size * val_pct)
    train_size = dataset_size - val_size
    train, val, test = random_split(dataset, [train_size, val_size, test_size])
    return train, val, test


dataloader_transforms = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
full_data = datasets.ImageFolder(root="../procData", transform=dataloader_transforms)
train_data, val_data, test_data = split_data(full_data)
trainLoader = DataLoader(train_data, batch_size=64, shuffle=True)
valLoader = DataLoader(val_data, batch_size=64, shuffle=True)
testLoader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(trainLoader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
