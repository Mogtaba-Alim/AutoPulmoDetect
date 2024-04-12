import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torchvision import models
import torchmetrics
from dataProcessing import DataLoaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Ensemble model is composite of GoogLeNet, ResNet-18 and DenseNet-121
class Ensemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Ensemble, self).__init__()
        self.googlenet = model1
        self.resnet = model2
        self.densenet = model3

    def forward(self, x):
        output1 = self.googlenet(x)
        output2 = self.resnet(x)
        output3 = self.densenet(x)
        outputs = (output1 + output2 + output3) / 3
        return outputs


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def finetune_base(model, dataloaders, crit, optimizer, num_epochs=25, model_name="ensemble"):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    # Metrics
    precision_metric = torchmetrics.Precision(num_classes=2, average='macro', task="binary").to(device)
    recall_metric = torchmetrics.Recall(num_classes=2, average='macro', task="binary").to(device)
    f1_metric = torchmetrics.F1Score(num_classes=2, average='macro', task="binary").to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

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
                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_precision = precision_metric.compute()
            epoch_recall = recall_metric.compute()
            epoch_f1 = f1_metric.compute()
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1 Score: {epoch_f1:.4f}')

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model along with the metrics
                torch.save({
                    'model_state': best_model_wts,
                    'precision': epoch_precision,
                    'recall': epoch_recall,
                    'f1_score': epoch_f1
                }, f"{model_name}_lr_{optimizer.param_groups[0]['lr']}_epochs_{num_epochs}_best_f1.pth")

    model.load_state_dict(best_model_wts)
    return model, best_f1

dataloader = DataLoaders()
criterion = nn.CrossEntropyLoss()

# Define hyperparameters
learning_rates = [0.001, 0.0005, 0.0001]
epochs_list = [25, 50, 75]

for lr in learning_rates:
    for epochs in epochs_list:
        print(f"Training ensemble with lr={lr}, epochs={epochs}")

        googlenet = models.googlenet(pretrained=True)
        freeze_params(googlenet)
        googlenet.fc = nn.Linear(in_features=1024, out_features=2)
        googlenet.to(device)

        resnet = models.resnet18(pretrained=True)
        freeze_params(resnet)
        resnet.fc = nn.Linear(in_features=512, out_features=2)
        resnet.to(device)

        densenet = models.densenet121(pretrained=True)
        freeze_params(densenet)
        densenet.classifier = nn.Linear(in_features=1024, out_features=2)
        densenet.to(device)

        ensemble_model = Ensemble(googlenet, resnet, densenet)
        optimizer = optim.Adam([
            {'params': googlenet.fc.parameters()},
            {'params': resnet.fc.parameters()},
            {'params': densenet.classifier.parameters()}
        ], lr=lr)

        trained_ensemble, best_f1 = finetune_base(ensemble_model, dataloader, criterion, optimizer, epochs, model_name="ensemble_model")
        print(f"Trained ensemble with lr={lr}, epochs={epochs}, Best F1={best_f1:.4f}")

