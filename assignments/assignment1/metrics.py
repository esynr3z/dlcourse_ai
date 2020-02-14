import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    num_samples = prediction.shape[0]

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = np.sum(prediction & ground_truth)
    tn = np.sum((~prediction) & (~ground_truth))
    fp = np.sum(prediction & (~ground_truth))
    fn = np.sum((~prediction) & ground_truth)

    accuracy = (tp + tn) / num_samples
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    num_samples = prediction.shape[0]
    correct = np.sum([prediction[i] == ground_truth[i] for i in range(num_samples)])
    accuracy = correct / num_samples

    return accuracy
