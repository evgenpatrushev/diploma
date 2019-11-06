import numpy as np


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.ones(nodes)
        self.input_shape, self.input, self.totals, self.out = (), np.array([]), np.array([]), np.array([])

    def forward(self, input_val):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.input_shape = input_val.shape

        input_val = input_val.flatten()
        self.input = input_val

        totals = np.dot(input_val, self.weights) + self.biases
        self.totals = totals

        exp = np.exp(totals)
        self.out = exp / np.sum(exp, axis=0)
        return self.out

    def backprop(self, d_l_d_out, learn_rate):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''

        exp = np.exp(self.totals)
        s = np.sum(exp)

        d_out_d_in = [(exp[i] * np.sum(list(exp[:i]) + list(exp[i + 1:]))) / (s ** 2) for i in range(len(exp))]
        self.weights = self.weights - learn_rate * np.dot(self.input[np.newaxis, ...].T,
                                                          (d_l_d_out * d_out_d_in)[np.newaxis, ...])
        self.biases = self.biases - learn_rate * d_l_d_out * d_out_d_in

        # previous version --------------------------------------------------------------------------------------------
        # We know only 1 element of d_L_d_out will be nonzero
        # for i, gradient in enumerate(d_L_d_out):
        #     if gradient == 0:
        #         continue
        #
        #     # e^totals
        #     t_exp = np.exp(self.last_totals)
        #
        #     # Sum of all e^totals
        #     S = np.sum(t_exp)
        #
        #     # Gradients of out[i] against totals
        #     d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
        #     d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
        #
        #     # Gradients of totals against weights/biases/input
        #     d_t_d_w = self.last_input
        #     d_t_d_b = 1
        #     d_t_d_inputs = self.weights
        #
        #     # Gradients of loss against totals
        #     d_L_d_t = gradient * d_out_d_t
        #
        #     # Gradients of loss against weights/biases/input
        #     d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
        #     d_L_d_b = d_L_d_t * d_t_d_b
        #     d_L_d_inputs = d_t_d_inputs @ d_L_d_t
        #
        #     # Update weights / biases
        #     self.weights -= learn_rate * d_L_d_w
        #     self.biases -= learn_rate * d_L_d_b

        # return d_L_d_inputs.reshape(self.last_input_shape)
        # -------------------------------------------------------------------------------------------------------------

        return (d_l_d_out * d_out_d_in).reshape(self.input_shape)
