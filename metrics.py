import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

accuracy = evaluate.load("accuracy")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy.compute(references=labels, predictions=preds)
    return {
        'accuracy': acc["accuracy"],
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
