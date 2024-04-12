import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torchvision import models

from dataProcessing import DataLoaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Ensemble model is composite of GoogLeNet, ResNet-18 and DenseNet-121
class Ensemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(self).__init__()
        self.googlenet = model1
        self.resnet = model2
        self.densenet = model3


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def finetune_base(model, dataloaders, crit, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Access dataloader using the get_loader method
            loader = dataloaders.get_loader(phase)

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = crit(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_acc


dataloader = DataLoaders()
criterion = nn.CrossEntropyLoss()

googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
freeze_params(googlenet)
googlenet.fc = nn.Linear(in_features=1024, out_features=2)
googlenet.to(device)
googlenetAdam = optim.Adam(googlenet.fc.parameters(), lr=0.001)
googlenetScheduler = optim.lr_scheduler.ReduceLROnPlateau(googlenetAdam, "min")
finetune_base(googlenet, dataloader, criterion, googlenetAdam, 30)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
freeze_params(resnet)
resnet.fc = nn.Linear(in_features=512, out_features=2)
resnet.to(device)
resnetAdam = optim.Adam(resnet.fc.parameters(), lr=0.001)
resnetScheduler = optim.lr_scheduler.ReduceLROnPlateau(resnetAdam, "min")
finetune_base(resnet, dataloader, criterion, resnetAdam, 30)

densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
freeze_params(densenet)
densenetClassifier = nn.Sequential(
  nn.Linear(in_features=2208, out_features=1024),
  nn.ReLU(),
  nn.Linear(in_features=1024, out_features=2)
)
densenet.classifier = densenetClassifier
densenet.to(device)
densenetAdam = optim.Adam(densenet.classifier.parameters(), lr=0.001)
densenetScheduler = optim.lr_scheduler.ReduceLROnPlateau(densenetAdam, "min")
finetune_base(densenet, dataloader, criterion, densenetAdam, 30)
