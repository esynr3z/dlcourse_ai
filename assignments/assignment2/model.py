import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = []
        self.layers.append(FullyConnectedLayer(n_input, hidden_layer_size))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(hidden_layer_size, n_output))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer in self.layers:
            layer.params_grad_clear()

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

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

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        l2_loss = 0.0
        for layer in self.layers:
            l2_loss += layer.params_l2_reg(self.reg)

        return loss + l2_loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        # Forward pass
        prev_layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(prev_layer_output)
            prev_layer_output = layer_output
        prob = softmax(layer_output)
        pred = np.argmax(prob, -1)

        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for layer_idx, layer in enumerate(self.layers):
            params = layer.params()
            for param_key in params:
                result[param_key + "_L%i" % layer_idx + "_%s" % layer.id] = params[param_key]

        return result
