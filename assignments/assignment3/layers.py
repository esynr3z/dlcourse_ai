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


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.X = None
        self.id = "CONV"

        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1  # stride is 1
        out_width = width - self.filter_size + 2 * self.padding + 1  # stride is 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        self.X = X.copy()
        p = self.padding
        X_padded = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), constant_values=0)

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        f = self.filter_size
        l_result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W_flat = self.W.value.reshape(-1, self.W.value.shape[-1])
        for y in range(out_height):
            for x in range(out_width):
                X_flat = X_padded[:, y:y + f, x:x + f, :].reshape(X_padded.shape[0], -1)  # flatten part of the input
                l_result[:, y, x] = np.dot(X_flat, W_flat) + self.B.value

        return l_result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        f = self.filter_size
        W_flat = self.W.value.reshape(-1, self.W.value.shape[-1])
        p = self.padding
        X_padded = np.pad(self.X, ((0, 0), (p, p), (p, p), (0, 0)), constant_values=0)
        d_padded = np.zeros_like(X_padded)
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                # Calculate W and B gradients
                X_flat = X_padded[:, y:y + f, x:x + f, :].reshape(X_padded.shape[0], -1)  # flatten part of the input
                self.B.grad += np.dot(np.ones((X_flat.shape[0], 1)).T, d_out[:, y, x, :]).reshape(self.B.grad.shape)
                self.W.grad += np.dot(X_flat.T, d_out[:, y, x, :]).reshape(self.W.grad.shape)

                # Calculate X gradients
                d_padded[:, y:y + f, x:x + f, :] += np.dot(d_out[:, y, x, :], W_flat.T).reshape(d_padded[:, y:y + f, x:x + f, :].shape)

        d_input = d_padded[:, p:-p, p:-p, :] if p else d_padded
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    def params_grad_clear(self):
        self.W.grad_clear()
        self.B.grad_clear()


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.id = "MAXPOOL"

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        s = self.stride
        f = self.pool_size

        out_height = ((height - f) // s) + 1
        out_width = ((width - f) // s) + 1

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        self.X = X.copy()

        l_result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                X_window = X[:, y * s:np.minimum(y * s + f, height), x * s:np.minimum(x * s + f, width), :]
                l_result[:, y, x, :] = X_window.max(axis=(1, 2))

        return l_result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        s = self.stride
        f = self.pool_size

        d_input = np.zeros_like(self.X)
        for b in range(batch_size):
            for c in range(channels):
                for y in range(out_height):
                    for x in range(out_width):
                        X_window = self.X[b, y * s:np.minimum(y * s + f, height), x * s:np.minimum(x * s + f, width), c]
                        d_input_argmax = np.unravel_index(np.argmax(X_window, axis=None), X_window.shape)
                        d_input_pool = np.zeros_like(X_window)
                        d_input_pool[d_input_argmax] = d_out[b, y, x, c]
                        d_input[b, y * s:np.minimum(y * s + f, height), x * s:np.minimum(x * s + f, width), c] += d_input_pool

        return d_input

    def params(self):
        return {}

    def params_grad_clear(self):
        # No params!
        pass


class Flattener:
    def __init__(self):
        self.X_shape = None
        self.id = "FLATTENER"

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}

    def params_grad_clear(self):
        # No params!
        pass
