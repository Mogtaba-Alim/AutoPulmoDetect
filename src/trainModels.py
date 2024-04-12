import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import time
import copy
import torchmetrics

from vggCustom import VGGCustom
from dataProcessing import DataLoaders


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    # Initialize metrics for binary classification
    precision = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)

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

            loader = dataloaders.get_loader(phase)

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1.update(preds, labels)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_precision = precision.compute()
            epoch_recall = recall.compute()
            epoch_f1 = f1.compute()
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1 Score: {epoch_f1:.4f}')

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_f1



def initialize_model(model_name, num_classes):
    if model_name == "vggcustom":
        model = VGGCustom()
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Model not supported")

    return model

def setup_and_train(model_name, num_classes, lr, num_epochs, dataloaders):
    print(f"Training {model_name} Model")
    model = initialize_model(model_name, num_classes)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model, best_f1 = train_model(model, dataloaders, criterion, optimizer, num_epochs)
    return model, best_f1

data = DataLoaders()

epochs_list = [25, 50, 75]
learning_rates = [0.001, 0.0005, 0.0001]

results = []

for epochs in epochs_list:
    for lr in learning_rates:
        print(f"Training with {epochs} epochs and learning rate {lr}")
        vgg_custom, vgg_best_f1 = setup_and_train("vggcustom", num_classes=2, lr=lr, num_epochs=epochs, dataloaders=data)
        torch.save(vgg_custom.state_dict(), f"vggcustom_lr_{lr}_epochs_{epochs}_best_model.pth")
        print(f"Saved VGGCustom with lr {lr} and epochs {epochs}, Best Validation F1-Score: {vgg_best_f1:.4f}")

        densenet, densenet_best_f1 = setup_and_train("densenet169", num_classes=2, lr=lr, num_epochs=epochs, dataloaders=data)
        torch.save(densenet.state_dict(), f"densenet169_lr_{lr}_epochs_{epochs}_best_model.pth")
        print(f"Saved DenseNet-169 with lr {lr} and epochs {epochs}, Best Validation F1-Score: {densenet_best_f1:.4f}")

        results.append((lr, epochs, 'VGGCustom', vgg_best_f1))
        results.append((lr, epochs, 'DenseNet-169', densenet_best_f1))

for result in results:
    print(f"Learning Rate: {result[0]}, Epochs: {result[1]}, Model: {result[2]}, Best F1-Score: {result[3]:.4f}")
