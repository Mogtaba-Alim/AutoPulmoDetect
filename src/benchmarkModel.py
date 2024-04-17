import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import copy
import torch.nn.functional as F

from dataProcessing import DataLoaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BenchmarkModel(nn.Module):
    def __init__(self):
        super(BenchmarkModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input channels = 3 (RGB), Output channels = 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Input channels = 32, Output channels = 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Input channels = 64, Output channels = 128
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer

        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)  # Input features = 128 (from global average pooling), Output features = 256
        self.fc2 = nn.Linear(256, 128)  # Input features = 256, Output features = 128
        self.fc3 = nn.Linear(128, 2)    # Input features = 128, Output features = 2 (number of classes)

    def forward(self, x):
        # Applying convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Applying global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Reduces each 128-channel feature map to 1x1
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Applying fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function here as this is the output layer
        return x

def finetune_base(model, dataloaders, crit, optimizer, num_epochs=25, model_name="model"):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
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
                torch.save(
                    best_model_wts,
                    f"{model_name}_lr_{optimizer.param_groups[0]['lr']}_epochs_{num_epochs}_best_f1.pth")

    model.load_state_dict(best_model_wts)
    return model, best_f1


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    dataloader = DataLoaders()
    criterion = nn.CrossEntropyLoss()

    # Define hyperparameters
    learning_rates = [0.001, 0.0005, 0.0001]
    epochs_list = [25, 50, 75]

    # Results list
    results = []

    # Hyperparameter tuning
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"Training Benchmark Model with lr={lr}, epochs={epochs}")
            benchmark_model = BenchmarkModel()
            benchmark_model.to(device)
            optimizer = optim.Adam(benchmark_model.parameters(), lr=lr)
            trained_benchmark, best_f1 = finetune_base(benchmark_model, dataloader, criterion, optimizer, epochs, model_name="benchmark_model")
            results.append((lr, epochs, best_f1))
            print(f"Trained Benchmark Model, lr={lr}, epochs={epochs}, Best F1={best_f1:.4f}")

    # Summarize results
    for result in results:
        print(f"Learning Rate: {result[0]}, Epochs: {result[1]}, Best F1-Score: {result[2]:.4f}")
