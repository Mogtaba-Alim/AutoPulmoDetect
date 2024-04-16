import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

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


def evaluate_model(model, loader):
    correct, total = 0, 0
    y, t = torch.Tensor(), torch.Tensor()
    true_negative, true_positive, false_negative, false_positive = 0, 0, 0, 0
    tn_hist, tp_hist, fn_hist, fp_hist = torch.zeros(10), torch.zeros(10), torch.zeros(10), torch.zeros(10)

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

        negative_label_mask = labels == 0
        positive_label_mask = labels == 1
        negative_pred_mask = preds == 0
        positive_pred_mask = preds == 1

        # true positive
        tp_batch_hist = torch.histc(outputs[positive_label_mask.logical_and(positive_pred_mask), 1], 10, min=0.5, max=1)
        tp_hist = tp_hist + tp_batch_hist

        # true negative
        tn_batch_hist = torch.histc(outputs[negative_label_mask.logical_and(negative_pred_mask), 0], 10, min=0.5, max=1)
        tn_hist = tn_hist + tn_batch_hist

        # false positive
        fp_batch_hist = torch.histc(outputs[negative_label_mask.logical_and(positive_pred_mask), 1], 10, min=0.5, max=1)
        fp_hist = fp_hist + fp_batch_hist

        # false negative
        fn_batch_hist = torch.histc(outputs[positive_label_mask.logical_and(negative_pred_mask), 0], 10, min=0.5, max=1)
        fn_hist = fn_hist + fn_batch_hist

    con_matrix = np.array([[true_negative, false_positive],
          [false_negative, true_positive]])
    return con_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total


def plot_hist(freqs, title, filename, skip_show=True):
    plt.figure()
    bins = np.linspace(0.5, 1, 10, endpoint=False)

    plt.title(title)
    plt.bar(bins, freqs, align='edge', width=0.045)
    plt.xticks(bins)
    plt.xlim(0.5, 1)
    plt.savefig(filename)

    if not skip_show:
        plt.show()


def plot_confusion_matrix(conf_matrix, title, filename, skip_show=True):
    cmp = ConfusionMatrixDisplay(conf_matrix, display_labels=["0", "1"])
    cmp.plot()
    plt.title(title)
    plt.savefig(filename)

    if not skip_show:
        plt.show()


def plot_graphs(cm, tp_h, tn_h, fp_h, fn_h, model_name):
    plot_hist(tp_h, f"{model_name} True Positive Confidence Histogram", f"../plots/{model_name}_TP_hist")
    plot_hist(tn_h, f"{model_name} True Negative Confidence Histogram", f"../plots/{model_name}_TN_hist")
    plot_hist(fp_h, f"{model_name} False Positive Confidence Histogram", f"../plots/{model_name}_FP_hist")
    plot_hist(fn_h, f"{model_name} False Negative Confidence Histogram", f"../plots/{model_name}_FN_hist")
    plot_confusion_matrix(cm, f"Confusion Matrix {model_name}", f"../plots/{model_name}_confusion")


os.makedirs(os.path.dirname("../plots/"), exist_ok=True)

data = DataLoaders()
loader = data.testLoader


## VGG EVALUATE
model = load_model("vggcustom", 2)
model.eval()

conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total = evaluate_model(model, loader)

print(f"VGG: correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
plot_graphs(conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, "VGG")


## DENSENET169 EVALUATE
model = load_model("densenet169", 2)
model.eval()

print("Loaded Model")
conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total = evaluate_model(model, loader)

print(f"Densenet169: correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
plot_graphs(conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, "Densenet169")
