import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import time
import copy

from vggCustom import VGGCustom
from dataProcessing import DataLoaders


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
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
                    loss = criterion(outputs, labels)

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
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model, best_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs)

    return model, best_acc


data = DataLoaders()

# Parameters for training
# Default
# learning_rate = 0.001
# epochs = 25
# Testing
learning_rate = 0.1
epochs = 1


# Train VGGCustom
vgg_custom, vgg_best_acc = setup_and_train("vggcustom", num_classes=2, lr=learning_rate, num_epochs=epochs, dataloaders=data)
torch.save(vgg_custom.state_dict(), "vggcustom_best_model.pth")
print(f"Saved VGGCustom with Best Validation Accuracy: {vgg_best_acc:.4f}")

# Train DenseNet-169
densenet, densenet_best_acc = setup_and_train("densenet169", num_classes=2, lr=learning_rate, num_epochs=epochs, dataloaders=data)
torch.save(densenet.state_dict(), "densenet169_best_model.pth")
print(f"Saved DenseNet-169 with Best Validation Accuracy: {densenet_best_acc:.4f}")
