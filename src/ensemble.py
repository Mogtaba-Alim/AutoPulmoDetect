import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torchmetrics
from torchvision import models

from dataProcessing import DataLoaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Ensemble model is composite of GoogLeNet, ResNet-18 and DenseNet-121

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def finetune_base(model, dataloaders, crit, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    A = []

    # Initialize metrics for binary classification
    precision = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)
    auc = torchmetrics.AUROC(num_classes=2, average='macro', task='binary').to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            precision.reset()
            recall.reset()
            f1.reset()
            auc.reset()

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
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1.update(preds, labels)
                auc.update(preds, labels)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_precision = precision.compute()
            epoch_recall = recall.compute()
            epoch_f1 = f1.compute()
            epoch_auc = auc.compute()
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1 Score: {epoch_f1:.4f} AUC Score: {epoch_auc:.4f}')

            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                A = [epoch_precision, epoch_recall, epoch_f1, epoch_auc]
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, A


dataloader = DataLoaders()
criterion = nn.CrossEntropyLoss()

googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
freeze_params(googlenet)
googlenet.fc = nn.Linear(in_features=1024, out_features=2)
googlenet.to(device)
googlenetAdam = optim.Adam(googlenet.fc.parameters(), lr=0.001)
googlenetScheduler = optim.lr_scheduler.ReduceLROnPlateau(googlenetAdam, "min")
# googlenet, A_googlenet = finetune_base(googlenet, dataloader, criterion, googlenetAdam, 5)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
freeze_params(resnet)
resnet.fc = nn.Linear(in_features=512, out_features=2)
resnet.to(device)
resnetAdam = optim.Adam(resnet.fc.parameters(), lr=0.001)
resnetScheduler = optim.lr_scheduler.ReduceLROnPlateau(resnetAdam, "min")
# resnet, A_resnet = finetune_base(resnet, dataloader, criterion, resnetAdam, 5)

densenet = models.densenet121(weights=models.DenseNet161_Weights.DEFAULT)
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
# densenet, A_densenet = finetune_base(densenet, dataloader, criterion, densenetAdam, 5)

# PLACEHOLDER A MATRICES
a1 = [0.9693, 0.9693, 0.9693, 0.9346]
a2 = [0.9775, 0.9693, 0.9734, 0.9483]
a3 = [0.9914, 0.9673, 0.9773, 0.9682]

w1 = sum([numpy.tanh(x) for x in a1])
w2 = sum([numpy.tanh(x) for x in a2])
w3 = sum([numpy.tanh(x) for x in a3])

print(w1)
print(w2)
print(w3)
