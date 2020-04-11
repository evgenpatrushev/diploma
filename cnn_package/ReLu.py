import numpy as np


class ReLU:
    # A standard fully-connected layer with ReLU activation.

    def __init__(self, input_len):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.normal(loc=0.1, scale=0.01, size=(input_len, input_len)) / (input_len * input_len)
        self.biases = np.ones(input_len)
        self.input_shape, self.input, self.totals, self.output = (), np.array([]), np.array([]), np.array([])

    def forward(self, input_val):
        """
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        """
        self.input_shape = input_val.shape

        input_val = input_val.flatten()
        self.input = input_val

        self.totals = np.dot(input_val, self.weights) + self.biases
        self.output = np.array([total if total >= 0 else 0 for total in self.totals]).reshape(self.input_shape)

        return self.output

    def backprop(self, d_l_d_out, learn_rate):
        """
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        """

        d_l_d_out = d_l_d_out.flatten()
        d_out_d_in = np.array([1 if total >= 0 else 0 for total in self.totals])

        d_l_d_x = np.dot(self.weights, (d_l_d_out * d_out_d_in)[..., np.newaxis]).reshape(self.input_shape)

        self.weights = self.weights - learn_rate * np.dot(self.input.T, d_l_d_out * d_out_d_in)
        self.biases = self.biases - learn_rate * d_l_d_out * d_out_d_in

        return d_l_d_x
