import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models, datasets
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from vggCustom import VGGCustom
from benchmarkModel import BenchmarkModel
from testOnlyDataProcessing import TestOnlyDataLoaders as DataLoaders
from sklearn.metrics import ConfusionMatrixDisplay

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
    elif model_name == "benchmark":
        model = BenchmarkModel()
        best_model_path = "./benchmark_model_lr_0.001_epochs_75_best_f1.pth"
    elif model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        best_model_path = "./resnet_Finetuned_lr_0.0005_epochs_75_best_model.pth"
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        best_model_path = "./googlenet_Finetuned_lr_0.001_epochs_75_best_model.pth"
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        best_model_path = "./densenet-121_Finetuned_lr_0.0005_epochs_75_best_model.pth"

    else:
        raise ValueError("Model not supported")

    model = model.to(device)

    # load the best model from file
    if device == "cpu":
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(best_model_path)

    model.load_state_dict(checkpoint)
    return model


def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    correct, total = 0, 0
    true_negative, false_positive, false_negative, true_positive = 0, 0, 0, 0  # Initialize here
    tn_hist, tp_hist, fn_hist, fp_hist = torch.zeros(10), torch.zeros(10), torch.zeros(10), torch.zeros(10)
    tn_hist, tp_hist, fn_hist, fp_hist = tn_hist.to(device), tp_hist.to(device), fn_hist.to(device), fp_hist.to(device)

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Confusion matrix updates
            true_negative += torch.sum((preds == 0) & (labels == 0)).item()
            false_positive += torch.sum((preds == 1) & (labels == 0)).item()
            false_negative += torch.sum((preds == 0) & (labels == 1)).item()
            true_positive += torch.sum((preds == 1) & (labels == 1)).item()

            # Histogram updates
            tp_mask = (labels == 1) & (preds == 1)
            tn_mask = (labels == 0) & (preds == 0)
            fp_mask = (labels == 0) & (preds == 1)
            fn_mask = (labels == 1) & (preds == 0)

            tp_hist += torch.histc(probabilities[tp_mask, 1], bins=10, min=0.5, max=1).to(device)
            tn_hist += torch.histc(probabilities[tn_mask, 0], bins=10, min=0.5, max=1).to(device)
            fp_hist += torch.histc(probabilities[fp_mask, 1], bins=10, min=0.5, max=1).to(device)
            fn_hist += torch.histc(probabilities[fn_mask, 0], bins=10, min=0.5, max=1).to(device)

            # Accuracy calculation
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    con_matrix = np.array([[true_negative, false_positive], [false_negative, true_positive]])

    # Convert histograms to numpy for return or further processing
    return con_matrix, tp_hist.cpu().numpy(), tn_hist.cpu().numpy(), fp_hist.cpu().numpy(), fn_hist.cpu().numpy(), correct, total



def ensemble_testing(dataloader, device, googlenet, resnet, densenet):
    w1 = sum([torch.tanh(x) for x in A_googlenet])
    w2 = sum([torch.tanh(x) for x in A_resnet])
    w3 = sum([torch.tanh(x) for x in A_densenet])

    googlenet.eval()
    resnet.eval()
    densenet.eval()

    # Initialize histogram tensors on the device
    tp_hist, tn_hist, fp_hist, fn_hist = torch.zeros(10, device=device), torch.zeros(10, device=device), \
        torch.zeros(10, device=device), torch.zeros(10, device=device)

    # Metrics initialization
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():  # Reduce memory usage by not storing gradients
        for i, t in dataloader:
            i = i.to(device)
            t = t.to(device)

            y_googlenet = torch.softmax(googlenet(i), dim=1)
            y_resnet = torch.softmax(resnet(i), dim=1)
            y_densenet = torch.softmax(densenet(i), dim=1)

            ensemble_preds = (y_googlenet * w1 + y_resnet * w2 + y_densenet * w3) / (w1 + w2 + w3)
            predictions = torch.argmax(ensemble_preds, dim=1)
            confidences = torch.max(ensemble_preds, dim=1).values

            # Confusion matrix updates
            tp += torch.sum((predictions == 1) & (t == 1)).item()
            tn += torch.sum((predictions == 0) & (t == 0)).item()
            fp += torch.sum((predictions == 1) & (t == 0)).item()
            fn += torch.sum((predictions == 0) & (t == 1)).item()

            # Histogram updates
            tp_mask = (t == 1) & (predictions == 1)
            tn_mask = (t == 0) & (predictions == 0)
            fp_mask = (t == 0) & (predictions == 1)
            fn_mask = (t == 1) & (predictions == 0)

            tp_hist += torch.histc(confidences[tp_mask], bins=10, min=0.5, max=1)
            tn_hist += torch.histc(confidences[tn_mask], bins=10, min=0.5, max=1)
            fp_hist += torch.histc(confidences[fp_mask], bins=10, min=0.5, max=1)
            fn_hist += torch.histc(confidences[fn_mask], bins=10, min=0.5, max=1)

    con_matrix = np.array([[tn, fp], [fn, tp]])
    correct = tp + tn
    total = tp + tn + fp + fn

    return con_matrix, tp_hist.cpu().numpy(), tn_hist.cpu().numpy(), fp_hist.cpu().numpy(), fn_hist.cpu().numpy(), correct, total



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
model = model.to(device)
model.eval()


conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total = evaluate_model(model, loader, device=device)

print(f"VGG: correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
plot_graphs(conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, "VGG")

del conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total, model

torch.cuda.empty_cache()


## DENSENET169 EVALUATE
model = load_model("densenet169", 2)
model = model.to(device)
model.eval()


print("Loaded Model")
conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total = evaluate_model(model, loader, device=device)

print(f"Densenet169: correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
plot_graphs(conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, "Densenet169")

torch.cuda.empty_cache()


## Benchmark EVALUATE
model = load_model("benchmark", 2)
model = model.to(device)
model.eval()

conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total = evaluate_model(model, loader, device=device)

print(f"Benchmark: correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
plot_graphs(conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, "Benchmark")

torch.cuda.empty_cache()
#
## Ensemble EVALUATE
resnet_model = load_model("resnet", 2)
resnet_model.eval()
googlenet_model = load_model("googlenet", 2)
googlenet_model.eval()

densenet_model = load_model("densenet121", 2)
densenet_model.eval()

A_resnet = torch.tensor([0.9680, 0.9737, 0.9708, 0.9432])
A_googlenet = torch.tensor([0.9626, 0.9795, 0.9710, 0.9382])
A_densenet = torch.tensor([0.9659, 0.9942, 0.9798, 0.9495])

conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, correct, total = ensemble_testing(loader, device=device,
                                                                                   googlenet=googlenet_model,
                                                                                   densenet=densenet_model,
                                                                                   resnet=resnet_model)

print(f"Ensemble: correct: {correct}, total: {total}, accuracy: {correct / total * 100:.02f}%")
plot_graphs(conf_matrix, tp_hist, tn_hist, fp_hist, fn_hist, "Ensemble")
