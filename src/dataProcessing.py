from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Create a DataLoaders class that will store the dataLoaders for train, validation and test, can be reused with other models.
class DataLoaders:
    def __init__(self):
        dataloader_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
        full_data = datasets.ImageFolder(root="../procData", transform=dataloader_transforms)
        train_data, val_data, test_data = self.split_data(full_data)
        self.trainLoader = DataLoader(train_data, batch_size=64, shuffle=True)
        self.valLoader = DataLoader(val_data, batch_size=64, shuffle=True)
        self.testLoader = DataLoader(test_data, batch_size=64, shuffle=True)

    def split_data(self, dataset):
        test_pct = 0.2
        val_pct = 0.1

        test_size = int(len(dataset) * test_pct)
        dataset_size = len(dataset) - test_size
        val_size = int(dataset_size * val_pct)
        train_size = dataset_size - val_size
        train, val, test = random_split(dataset, [train_size, val_size, test_size])
        return train, val, test


data = DataLoaders()
train_features, train_labels = next(iter(data.trainLoader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].permute(1, 2, 0).squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
