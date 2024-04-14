from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Create a DataLoaders class that will store the dataLoaders for train, validation and test, can be reused with other models.
class TestOnlyDataLoaders:
    def __init__(self):
        dataloader_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
        full_data = datasets.ImageFolder(root="../testData/procData", transform=dataloader_transforms)

        self.testLoader = DataLoader(full_data, batch_size=64, shuffle=True)


data = TestOnlyDataLoaders()
# train_features, train_labels = next(iter(data.trainLoader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].permute(1, 2, 0).squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
