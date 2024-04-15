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

    cm = np.array([[true_negative, false_positive],
          [false_negative, true_positive]])
    print(f"correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
    return cm, tp_hist, tn_hist, fp_hist, fn_hist


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


cm, tp_h, tn_h, fp_h, fn_h = evaluate_model(model, loader)
os.makedirs(os.path.dirname("../plots/"), exist_ok=True)

plot_hist(tp_h, "VGG True Positive Confidence Histogram", "../plots/vgg_TP_hist")
plot_hist(tn_h, "VGG True Negative Confidence Histogram", "../plots/vgg_TN_hist")
plot_hist(fp_h, "VGG False Positive Confidence Histogram", "../plots/vgg_FP_hist")
plot_hist(fn_h, "VGG False Negative Confidence Histogram", "../plots/vgg_FN_hist")
plot_confusion_matrix(cm, "Confusion Matrix VGG", "../plots/vgg_confusion")
