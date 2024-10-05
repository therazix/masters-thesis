import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(labels: np.ndarray, predictions: np.ndarray):
    """
    Compute metrics for classification tasks
    :param labels: 1D array of true labels
    :param predictions: 1D array of predicted labels
    :return: dictionary of metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
