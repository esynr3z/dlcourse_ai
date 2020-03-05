import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layers = []

        image_width, image_height, n_channels = input_shape
        filter_size = 3
        pool_size = 4
        stride = pool_size
        padding = 1
        fc_input = (image_height // (pool_size ** 2)) * (image_width // (pool_size ** 2)) * conv2_channels

        self.layers.append(ConvolutionalLayer(n_channels, conv1_channels, filter_size, padding))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size, stride))

        self.layers.append(ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size, stride))

        self.layers.append(Flattener())
        self.layers.append(FullyConnectedLayer(fc_input, n_output_classes))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for layer in self.layers:
            layer.params_grad_clear()

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        # Forward pass
        prev_layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(prev_layer_output)
            prev_layer_output = layer_output
        loss, loss_grad = softmax_with_cross_entropy(layer_output, y)

        # Backward pass
        prev_layer_grad = loss_grad
        for layer in reversed(self.layers):
            layer_grad = layer.backward(prev_layer_grad)
            prev_layer_grad = layer_grad

        return loss

    def predict(self, X):
        # Forward pass
        prev_layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(prev_layer_output)
            prev_layer_output = layer_output

        prob = softmax(layer_output)
        pred = np.argmax(prob, axis=-1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for layer_idx, layer in enumerate(self.layers):
            params = layer.params()
            for param_key in params:
                result[param_key + "_L%i" % layer_idx + "_%s" % layer.id] = params[param_key]

        return result
