import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.power(W, 2))
    grad = W * 2 * reg_strength

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    if (predictions.ndim == 1):
        predictions_stab = predictions - np.max(predictions)
        prob = np.exp(predictions_stab) / np.sum(np.exp(predictions_stab))
    else:  # for batches
        predictions_stab = predictions - np.max(predictions, -1)[:, None]
        prob = np.exp(predictions_stab) / np.sum(np.exp(predictions_stab), -1)[:, None]

    return prob


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    with np.errstate(divide="ignore"):
        if (probs.ndim == 1):
            loss = -np.log(probs[target_index][0])
        else:  # for batches
            loss = np.mean(-np.log(probs[np.arange(probs.shape[0]), target_index]))
    if np.isinf(loss):
        loss = 0

    return loss


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    if (preds.ndim == 1):
        batch_size = 1
    else:
        batch_size = preds.shape[0]
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    grad = probs.copy()
    if (grad.ndim == 1):
        grad[target_index] -= 1
    else:  # for batches
        grad[np.arange(batch_size), target_index] -= 1
    grad = grad / batch_size

    return loss, grad


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = None
        self.grad_clear()

    def grad_clear(self):
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        self.X = None
        self.id = "RELU"

    def forward(self, X):
        # TODO_: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        result = np.maximum(X, 0)
        self.X = X
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO_: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_X = (self.X > 0) * d_out
        return d_X

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

    def params_grad_clear(self):
        # ReLU Doesn't have any parameters
        pass

    def params_l2_reg(self, reg):
        # ReLU Doesn't have any parameters
        return 0.0


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.id = "FC"

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()

        l_result = np.dot(X, self.W.value) + self.B.value

        return l_result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # Calculate W and B gradients
        self.B.grad = np.dot(np.ones((self.X.shape[0], 1)).T, d_out)
        self.W.grad = np.dot(self.X.T, d_out)

        # Calculate X gradients
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    def params_grad_clear(self):
        self.W.grad_clear()
        self.B.grad_clear()

    def params_l2_reg(self, reg):
        l2_loss = 0.0

        loss, grad = l2_regularization(self.W.value, reg)
        l2_loss += loss
        self.W.grad += grad

        loss, grad = l2_regularization(self.B.value, reg)
        l2_loss += loss
        self.B.grad += grad

        return l2_loss
