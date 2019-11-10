import numpy as np

np.seterr(all='raise')


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.normal(loc=0, scale=5, size=(input_len, nodes)) / input_len
        # self.weights = np.random.normal(loc=0, scale=0.2, size=(input_len, nodes))
        self.biases = np.zeros(nodes)
        self.input_shape, self.input, self.totals, self.out = (), np.array([]), np.array([]), np.array([])

    def forward(self, input_val):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.input_shape = input_val.shape

        input_val = input_val.flatten() / np.max(input_val)
        # input_val = input_val.flatten()
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

        for i, gradient in enumerate(d_l_d_out):
            if gradient == 0:
                continue

            exp = np.exp(self.totals)
            s = np.sum(exp)

            # Gradients of out[i] against totals
            d_out_d_in = - exp[i] * exp / (s ** 2)
            d_out_d_in[i] = exp[i] * (s - exp[i]) / (s ** 2)

            d_l_d_x = np.dot(self.weights, (gradient * d_out_d_in)[..., np.newaxis]).reshape(self.input_shape)

            self.weights = self.weights - learn_rate * np.dot(self.input[..., np.newaxis],
                                                              (gradient * d_out_d_in)[np.newaxis, ...])

            self.biases = self.biases - learn_rate * gradient * d_out_d_in

            return d_l_d_x

        exp = np.exp(self.totals)
        s = np.sum(exp)

        d_out_d_in = [(exp[i] * np.sum(list(exp[:i]) + list(exp[i + 1:]))) / (s ** 2) for i in range(len(exp))]
        self.weights = self.weights - learn_rate * np.dot(self.input[..., np.newaxis],
                                                          (gradient * d_out_d_in)[np.newaxis, ...])
        self.biases = self.biases - learn_rate * d_l_d_out * d_out_d_in
        return np.dot(self.weights, (d_l_d_out * d_out_d_in)[..., np.newaxis]).reshape(self.input_shape)


if __name__ == '__main__':
    obj = Softmax(3, 3)
    obj.weights = np.array([[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.5, 0.2, 0.9]])
    obj.forward(input_val=np.array([0.938, 0.94, 0.98]))
    label = np.array([1, 0, 0])
    obj.out = np.array([0.26980, 0.32235, 0.40784])
    d_l_d_out = -1 * label / obj.out - (1 - label) / (1 - obj.out)
    obj.backprop(d_l_d_out, 0.01)
