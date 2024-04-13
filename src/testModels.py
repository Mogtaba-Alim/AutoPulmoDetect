import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from vggCustom import VGGCustom
from testOnlyDataProcessing import TestOnlyDataLoaders as DataLoaders


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_name, num_classes):
    # initialize model
    if model_name == "vggcustom":
        model = VGGCustom()
        best_model_path = "./vggcustom_lr_0.0001_epochs_75_best_model.pth"
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        best_model_path = "./densenet169_lr_0.0001_epochs_75_best_model.pth"

    else:
        raise ValueError("Model not supported")

    # load the best model from file
    if device == "cpu":
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(best_model_path)

    model.load_state_dict(checkpoint)
    return model


data = DataLoaders()

epochs_list = [25, 50, 75]
learning_rates = [0.001, 0.0005, 0.0001]

results = []

model = load_model("vggcustom", 2)

model.eval()

loader = data.testLoader


def evaluate_model(model, loader):
    correct, total = 0, 0
    y, t = torch.Tensor(), torch.Tensor()
    true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0

    for inputs, labels in loader:
        # Calculating predictions and outputs of Model
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Accuracy information
        correct += int(torch.sum(preds == labels))
        total += labels.shape[0]

        # Confusion matrix information
        cm = confusion_matrix(labels, preds)
        true_negative += cm[0][0]
        false_positive += cm[0][1]
        false_negative += cm[1][0]
        true_positive += cm[1][1]

        # Histogram information
        outputs = torch.softmax(outputs, 1)

    cm = np.array([[true_negative, false_positive],
          [false_negative, true_positive]])
    print(f"correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
    return cm, 1


cm, t = evaluate_model(model, loader)
cmp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
cmp.plot()
plt.title("Confusion Matrix (Healthy vs Diseased Data)")
plt.show()
