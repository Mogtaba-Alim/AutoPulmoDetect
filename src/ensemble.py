import torch
import torch.nn as nn
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
    best_f1 = 0.0
    A = []

    # Initialize metrics for binary classification
    precision = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)
    auc = torchmetrics.AUROC(num_classes=2, average='macro', task='binary').to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
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
            print(
                f'{phase.capitalize()} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1 Score: {epoch_f1:.4f} AUC Score: {epoch_auc:.4f}')

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                A = [epoch_precision, epoch_recall, epoch_f1, epoch_auc]
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, A


def ensemble_testing(type: int):
    w1 = sum([torch.tanh(x) for x in A_googlenet])
    w2 = sum([torch.tanh(x) for x in A_resnet])
    w3 = sum([torch.tanh(x) for x in A_densenet])

    p1 = torch.empty((0, 2)).to(device)
    p2 = torch.empty((0, 2)).to(device)
    p3 = torch.empty((0, 2)).to(device)
    true_vals = torch.empty(0).to(device)
    googlenet.eval()
    resnet.eval()
    densenet.eval()
    for i, t in dataloader.testLoader:
        i = i.to(device)
        t = t.to(device)
        y_googlenet = googlenet(i)
        p1 = torch.cat((p1, torch.softmax(y_googlenet, dim=1)))
        y_resnet = resnet(i)
        p2 = torch.cat((p2, torch.softmax(y_resnet, dim=1)))
        y_densenet = densenet(i)
        p3 = torch.cat((p3, torch.softmax(y_densenet, dim=1)))
        true_vals = torch.cat((true_vals, t))

    correct = 0
    for j in range(p1.shape[0]):
        ensemble_j = (p1[j] * w1 + p2[j] * w2 + p3[j] + w3) / (w1 + w2 + w3)
        prediction = torch.argmax(ensemble_j)
        if prediction == true_vals[j]:
            correct += 1

    print(f'Accuracy: {correct / p1.shape[0]:.4f}')

    if type == 1:
        # Write ensemble accuracy to file
        with open('hyperParameterTuningResults.txt', 'a') as file:
            file.write('Default Paper Hyperparameters Ensemble Model Performance:\n')
            file.write(f'Accuracy: {correct / p1.shape[0]:.4f}\n\n')
    elif type == 2:
        # Write ensemble accuracy to file
        with open('hyperParameterTuningResults.txt', 'a') as file:
            file.write(' Hyperparameter Finetuning Ensemble Model Performance:\n')
            file.write(f'Accuracy: {correct / p1.shape[0]:.4f}\n\n')


dataloader = DataLoaders()
criterion = nn.CrossEntropyLoss()

googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
freeze_params(googlenet)
googlenet.fc = nn.Linear(googlenet.fc.in_features, out_features=2)
googlenet.to(device)
googlenetAdam = optim.Adam(googlenet.fc.parameters(), lr=0.001)
googlenet, A_googlenet = finetune_base(googlenet, dataloader, criterion, googlenetAdam, 2)
torch.save(googlenet.state_dict(), f"googlenet_default_lr_{0.001}_epochs_{25}_best_model.pth")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
freeze_params(resnet)
resnet.fc = nn.Linear(resnet.fc.in_features, out_features=2)
resnet.to(device)
resnetAdam = optim.Adam(resnet.fc.parameters(), lr=0.001)
resnet, A_resnet = finetune_base(resnet, dataloader, criterion, resnetAdam, 2)
torch.save(resnet.state_dict(), f"resnet_default_lr_{0.001}_epochs_{25}_best_model.pth")

densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
freeze_params(densenet)
densenet.classifier = nn.Linear(in_features=1024, out_features=2)
densenet.to(device)
densenetAdam = optim.Adam(densenet.classifier.parameters(), lr=0.001)
densenet, A_densenet = finetune_base(densenet, dataloader, criterion, densenetAdam, 2)
torch.save(densenet.state_dict(), f"densenet-121_default_lr_{0.001}_epochs_{25}_best_model.pth")

# PLACEHOLDER A MATRICES
# a1 = [0.9693, 0.9693, 0.9693, 0.9346]
# a2 = [0.9775, 0.9693, 0.9734, 0.9483]
# a3 = [0.9914, 0.9673, 0.9773, 0.9682]
# w1 = sum([np.tanh(x) for x in a1])
# w2 = sum([np.tanh(x) for x in a2])
# w3 = sum([np.tanh(x) for x in a3])


ensemble_testing(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def finetune_base(model, dataloaders, crit, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = {}
    best_f1 = 0.0

    precision = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device)
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)
    auc = torchmetrics.AUROC(num_classes=2, average='macro', task='binary').to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            precision.reset()
            recall.reset()
            f1.reset()
            auc.reset()

            for inputs, labels in dataloaders.get_loader(phase):
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

            if phase == 'val':
                epoch_f1 = f1.compute()
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_metrics = {
                        'loss': running_loss / len(dataloaders.get_loader(phase).dataset),
                        'precision': precision.compute(),
                        'recall': recall.compute(),
                        'f1': epoch_f1,
                        'auc': auc.compute()
                    }
                    best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_metrics


def train_and_select_best_model(model, ep_list, lear_rates, dataloaders, name):
    best_model = None
    best_metrics = None
    best_f1 = 0.0
    best_params = None
    for lr in lear_rates:
        for epochs in ep_list:
            temp_model = copy.deepcopy(model)
            temp_model.to(device)
            freeze_params(temp_model)

            # Modify the classifier according to the model type
            if name == 'DenseNet-121':
                num_ftrs = temp_model.classifier.in_features
                temp_model.classifier = nn.Linear(num_ftrs, 2)
            else:  # Assuming 'GoogLeNet' or 'ResNet-18'
                num_ftrs = temp_model.fc.in_features
                temp_model.fc = nn.Linear(num_ftrs, 2)

            temp_model.to(device)
            optimizer = optimizer = optim.Adam(filter(lambda p: p.requires_grad, temp_model.parameters()), lr=lr)
            trained_model, metrics = finetune_base(temp_model, dataloaders, criterion, optimizer, epochs)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model = trained_model
                best_metrics = metrics
                best_params = [lr, epochs]

    # Write the best model metrics to file
    with open('hyperParameterTuningResults.txt', 'a') as file:
        file.write(f'Best {name} Model Performance:\n')
        file.write(f'Learning Rate: {best_params[0]}, Epochs: {best_params[1]}\n')
        file.write(
            f"Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1 Score: {best_metrics['f1']:.4f}, AUC: {best_metrics['auc']:.4f}\n\n")
    return best_model, [best_metrics['precision'], best_metrics['recall'], best_metrics['f1'],
                        best_metrics['auc']], best_params




dataloader = DataLoaders()
criterion = nn.CrossEntropyLoss()


# Hyperparameters
epochs_list = [25, 50, 75]
learning_rates = [0.001, 0.0005, 0.0001]

# Train and select the best models
googlenet, A_googlenet, googlenet_params = train_and_select_best_model(
    models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT), epochs_list, learning_rates, dataloader, 'GoogLeNet')
torch.save(googlenet.state_dict(),
           f"googlenet_Finetuned_lr_{googlenet_params[0]}_epochs_{googlenet_params[1]}_best_model.pth")
resnet, A_resnet, resnet_params = train_and_select_best_model(models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                                                              epochs_list, learning_rates, dataloader, 'ResNet-18')
torch.save(resnet.state_dict(), f"resnet_Finetuned_lr_{resnet_params[0]}_epochs_{resnet_params[1]}_best_model.pth")
densenet, A_densenet, densenet_params = train_and_select_best_model(
    models.densenet121(weights=models.DenseNet121_Weights.DEFAULT), epochs_list, learning_rates, dataloader,
    'DenseNet-121')
torch.save(densenet.state_dict(),
           f"densenet-121_Finetuned_lr_{densenet_params[0]}_epochs_{densenet_params[1]}_best_model.pth")

ensemble_testing(2)
