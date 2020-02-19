import numpy as np


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
