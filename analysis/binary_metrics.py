import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC


def calculate_metrics(task, outputs, labels):
    # metrics:
    accuracy = Accuracy(task=task)
    precision = Precision(task=task)
    recall = Recall(task=task)
    auroc = AUROC(task=task)
    metrics = MetricCollection([accuracy, precision, recall, auroc])
    # convert labels to int
    labels = labels.type(torch.int)
    metrics(outputs, labels)
    return {
        "Accuracy": accuracy.compute(),
        "Sensitivity": recall.compute(),
        "PPV": precision.compute(),
        "AUC": auroc.compute(),
    }


def calculate_np_metrics(outputs: Tensor | np.ndarray, labels: Tensor | np.ndarray):
    if isinstance(outputs, Tensor):
        outputs = outputs.numpy()
        labels = labels.numpy()

    if labels.ndim == 2:
        labels = labels.flatten()
        outputs = outputs.flatten()

    # calculate the AUC
    if len(set(labels)) == 1:
        auc = 0
    elif len(set(labels)) > 2:
        # binarize the labels
        labels = np.where(labels > 0.51, 1, 0)
        auc = roc_auc_score(labels, outputs)
    else:
        auc = roc_auc_score(labels, outputs)

    # threshold the outputs
    outputs = np.where(outputs > 0.5, 1, 0)

    metrics = base_calculate_metrics(labels, outputs)
    metrics["AUC"] = auc

    return metrics


def show_metrics(metrics: dict):
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def base_calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # calculate the AUC
    if len(set(y_true)) == 1:
        auc = 0
    else:
        auc = roc_auc_score(y_true, y_pred)

    # calculate the metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if tp + fn != 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp != 0 else 0.0
    ppv = tp / (tp + fp) if tp + fp != 0 else 0.0
    npv = tn / (tn + fn) if tn + fn != 0 else 0.0

    # calculate F1 score
    f1_score = (
        2 * (ppv * sensitivity) / (ppv + sensitivity)
        if (ppv + sensitivity) != 0
        else 0.0
    )

    # calculate Kappa coefficient
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / ((tp + tn + fp + fn) ** 2)
    kappa = (accuracy - pe) / (1 - pe)

    # calculate MCC coefficient
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (
        (tp * tn - fp * fn) / (mcc_denominator**0.5) if mcc_denominator != 0 else 0.0
    )

    metrics = {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "SEN": sensitivity,
        "SPEC": specificity,
        "PPV": ppv,
        "NPV": npv,
        "AUC": auc,
        "Acc": accuracy,
        "BAcc": (sensitivity + specificity) / 2,
        "F1": f1_score,
        "Kappa": kappa,
        "MCC": mcc,
    }

    return metrics
