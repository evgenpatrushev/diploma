import numpy as np

np.seterr(all='raise')


class Softmax:

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights, self.grad_weights = np.random.normal(loc=0, scale=5, size=(input_len, nodes)) / input_len, \
                                          np.zeros((input_len, nodes))
        # self.weights = np.random.normal(loc=0, scale=0.2, size=(input_len, nodes))
        self.biases, self.grad_biases = np.zeros(nodes), np.zeros(nodes)

        self.first_momentum_dw, self.first_momentum_db = np.zeros(self.weights.shape), np.zeros(self.biases.shape)
        self.second_momentum_dw, self.second_momentum_db = np.zeros(self.weights.shape), np.zeros(self.biases.shape)

        self.input_shape, self.input, self.totals, self.output = (), np.array([]), np.array([]), np.array([])

    def forward(self, input_val):
        """
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        """
        self.input_shape = input_val.shape

        # input_val = input_val.flatten()
        if np.max(input_val) > 1:
            self.input = input_val.flatten() / np.max(input_val)
        else:
            self.input = input_val.flatten()

        totals = np.dot(self.input, self.weights) + self.biases
        self.totals = totals

        exp = np.exp(totals)
        exp_sum = np.sum(exp)
        self.output = np.zeros(exp.shape)

        def exception_check(exponent, s, index):
            try:
                self.output[index] = exponent / s
            except FloatingPointError:
                if abs(exponent) < abs(s):
                    self.output[index] = 0
                else:
                    raise Exception('exception_check error of softmax function diviation')

        for i, e in enumerate(exp):
            exception_check(exponent=e, index=i, s=exp_sum)

        return self.output

    def backprop(self, d_l_d_out):
        """
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        """

        for i, gradient in enumerate(d_l_d_out):
            if gradient == 0:
                continue

            exp = np.exp(self.totals)
            s = np.sum(exp)

            # Gradients of out[i] against totals
            d_out_d_in = - exp[i] * exp / (s ** 2)
            d_out_d_in[i] = exp[i] * (s - exp[i]) / (s ** 2)

            d_l_d_x = np.dot(self.weights, (gradient * d_out_d_in)[..., np.newaxis]).reshape(self.input_shape)

            self.grad_weights = np.dot(self.input[..., np.newaxis], (gradient * d_out_d_in)[np.newaxis, ...])

            self.grad_biases = gradient * d_out_d_in

            return d_l_d_x

        raise RuntimeError("Fully zero dL/d_out")

    def update_weights(self, learning_rate=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8,
                       correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        if optimization != 'adam':
            self.weights -= learning_rate * (self.grad_weights + l2_penalty * self.weights)
            self.biases -= learning_rate * (self.grad_biases + l2_penalty * self.biases)

        else:
            if correct_bias:
                w_first_moment = self.first_momentum_dw / (1 - beta1 ** iter)
                b_first_moment = self.first_momentum_db / (1 - beta1 ** iter)
                w_second_moment = self.second_momentum_dw / (1 - beta2 ** iter)
                b_second_moment = self.second_momentum_db / (1 - beta2 ** iter)
            else:
                w_first_moment = self.first_momentum_dw
                b_first_moment = self.first_momentum_db
                w_second_moment = self.second_momentum_dw
                b_second_moment = self.second_momentum_db

            w_learning_rate = learning_rate / (np.sqrt(w_second_moment) + epsilon)
            b_learning_rate = learning_rate / (np.sqrt(b_second_moment) + epsilon)

            self.weights -= w_learning_rate * (w_first_moment + l2_penalty * self.weights)
            self.biases -= b_learning_rate * (b_first_moment + l2_penalty * self.biases)

    def first_momentum(self, beta=0.9):
        self.first_momentum_dw = beta * self.first_momentum_dw + (1 - beta) * self.grad_weights
        self.first_momentum_db = beta * self.first_momentum_db + (1 - beta) * self.grad_biases

    def second_momentum(self, beta=0.999, amsprop=False):
        new_dw = beta * self.second_momentum_dw + (1 - beta) * (self.grad_weights ** 2)
        new_db = beta * self.second_momentum_db + (1 - beta) * (self.grad_biases ** 2)

        if amsprop:
            self.second_momentum_dw = np.maximum(self.second_momentum_dw, new_dw)
            self.second_momentum_db = np.maximum(self.second_momentum_db, new_db)
        else:
            self.second_momentum_dw = new_dw
            self.second_momentum_db = new_db


if __name__ == '__main__':
    obj = Softmax(3, 3)
    obj.weights = np.array([[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.5, 0.2, 0.9]])
    obj.forward(input_val=np.array([0.938, 0.94, 0.98]))
    label = np.array([1, 0, 0])
    obj.output = np.array([0.26980, 0.32235, 0.40784])
    d_l_d_out = -1 * label / obj.output - (1 - label) / (1 - obj.output)
    obj.backprop(d_l_d_out)
