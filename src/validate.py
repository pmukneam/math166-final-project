# This file contains the functions for evaluating the accuracy, precision, recall and F1 score of the given predictions.
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def evaluate_score(y_true, y_pred):
    """
    Evaluate the accuracy, precision, recall and F1 score of the given predictions.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy, precision, recall and F1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
