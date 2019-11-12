import numpy as np


class Convolution:
    # A Convolution layer using (shape x shape) filters.

    def __init__(self, num_filters=1, shape=3, stride=1, padding=0, image=False):
        self.image = image
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.shape = shape

        # self.filters = np.random.normal(loc=0, scale=0.2, size=(num_filters, shape, shape))
        self.filters, self.grad_filters = np.random.normal(loc=0, scale=5, size=(num_filters, shape, shape)) / (
                self.num_filters + self.shape + self.shape), np.zeros((num_filters, shape, shape))

        # self.biases, self.grad_biases = np.zeros(num_filters), np.zeros(num_filters)
        self.input, self.output = np.array([]), np.array([])

    def padding_f(self, matrix, padding):
        h, w = matrix.shape
        res = np.zeros((h + 2 * padding, w + 2 * padding))
        res[padding:-padding, padding:-padding] = matrix
        return res

    def iterate_regions(self, image):
        '''
        Generates all possible (shape x shape) image regions using valid padding.
        - image is a 2d numpy array.
        '''
        h, w = image.shape

        regions = ((i, j, image[i:(i + self.shape), j:(j + self.shape)])
                   for i in range(0, h - self.shape + 1, self.stride)
                   for j in range(0, w - self.shape + 1, self.stride))

        for i, j, region in regions:
            yield i, j, region

    def forward(self, input_val):
        if self.image:

            if len(input_val.shape) == 2:
                self._forward(input_val=input_val)
                # self.output = self.output + self.biases
                self.output = self.output

            elif len(input_val.shape) == 3:
                """
                RGB images are splitting into three color layers and for each color are created filter and then 
                sum into one output (image/conv) with bias.
                
                Combination of 3 filters are created num_filters times
                
                Algorithm: 
                    1 - get 3 color RGB layers
                    2 - random filters for each RGB layer created 'num_filters' filters with shape 'shape*shape'
                    3 - by num_filters convoluted each RGB layer and then sum for each RGB into one 
                    4 - get np array with shape (shape, shape, num_filters)   
                """

                filters = np.random.normal(loc=0, scale=5, size=(3, self.num_filters, self.shape, self.shape)) / (
                        self.num_filters + self.shape + self.shape)
                # filters = np.random.normal(loc=0, scale=0.2, size=(3, self.num_filters, self.shape, self.shape))
                output = 0

                for img_color, filter_for_color in zip(input_val, filters):
                    self.filters = filter_for_color
                    self._forward(input_val=img_color)
                    output += self.output

                self.filters = filters
                # self.output = output + self.biases
                self.output = output

        else:
            num_conv = input_val.shape[2]
            output = 0
            for i in range(num_conv):
                self._forward(input_val=input_val[:, :, i])
                # self.output += self.biases
                if not i:  # first iteration i == 0
                    output = self.output
                else:
                    output = np.concatenate((output, self.output), axis=2)
            self.output = output

        self.input = input_val

    def _forward(self, input_val):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''

        if self.padding:
            input_val = self.padding_f(input_val, self.padding)

        h, w = input_val.shape
        self.output = np.zeros(
            ((h - self.shape) // self.stride + 1,
             (w - self.shape) // self.stride + 1, self.num_filters))

        for i, j, im_region in self.iterate_regions(input_val):
            self.output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return self.output

    def back_propagation(self, d_l_d_out, learn_rate):

        # TODO assert if d_l_d_out shape != input shape

        if self.image:

            if len(self.input.shape) == 2:
                self._backprop(d_l_d_out=d_l_d_out, learn_rate=learn_rate, input_val=self.input,
                               d_l_d_f=self.grad_filters)

            elif len(self.input.shape) == 3:
                # TODO -remake
                filters = self.filters
                for i, img_color, filters_for_color in enumerate(zip(self.input, filters)):
                    self.filters = filters_for_color
                    self._backprop(d_l_d_out=d_l_d_out, learn_rate=learn_rate, input_val=img_color)
                    filters[i] = self.filters

        else:
            """
            d_L_d_out have to be matrix same size as output of forward function 
            but with error for backprop
            """

            num_conv = self.input.shape[2]
            return_val = np.zeros(self.input.shape)

            for i in range(num_conv):
                return_val[:, :, i] = self._backprop(
                    d_l_d_out=d_l_d_out[:, :, i * self.num_filters:i * self.num_filters + self.num_filters],
                    input_val=self.input[:, :, i], d_l_d_f=self.grad_filters, learn_rate=learn_rate)

            return return_val

    def _backprop(self, d_l_d_out, input_val, d_l_d_f, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''

        # Create return val if it's not first layer
        return_val = None
        if not self.image:
            d_l_d_x = np.zeros(input_val.shape)

            padding = self.shape - 1 - self.padding

            # TODO -check this part of logic
            for f in range(self.num_filters):

                if padding:
                    iter_val = self.padding_f(d_l_d_out[:, :, f], padding)
                else:
                    iter_val = d_l_d_out[:, :, f]

                w = np.rot90(self.filters[f], 2)

                for i, j, im_region in self.iterate_regions(iter_val):
                    d_l_d_x[i, j] += np.sum(im_region * w)
            return_val = d_l_d_x

        # Update filters and biases

        if self.padding:
            input_val = self.padding_f(input_val, self.padding)

        for i, j, im_region in self.iterate_regions(input_val):
            for f in range(self.num_filters):
                d_l_d_f[f] += d_l_d_out[i, j, f] * im_region

        # self.biases -= np.sum(d_l_d_out, axis=(0, 1))
        # TODO -add biases update

        return return_val

    def update_weights(self, learning_rate=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8,
                       correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        if optimization != 'adam':
            self.filters -= learning_rate * (self.grad_filters + l2_penalty * self.filters)
            # self.biases -= learning_rate * (self.grads_biases + l2_penalty * self.biases)

        else:
            if correct_bias:
                w_first_moment = self.momentum_cache['dW'] / (1 - beta1 ** iter)
                # b_first_moment = self.momentum_cache['db'] / (1 - beta1 ** iter)
                w_second_moment = self.rmsprop_cache['dW'] / (1 - beta2 ** iter)
                # b_second_moment = self.rmsprop_cache['db'] / (1 - beta2 ** iter)
            else:
                w_first_moment = self.momentum_cache['dW']
                # b_first_moment = self.momentum_cache['db']
                w_second_moment = self.rmsprop_cache['dW']
                # b_second_moment = self.rmsprop_cache['db']

            w_learning_rate = learning_rate / (np.sqrt(w_second_moment) + epsilon)
            # b_learning_rate = learning_rate / (np.sqrt(b_second_moment) + epsilon)

            self.filters -= w_learning_rate * (w_first_moment + l2_penalty * self.filters)
            # self.params['b'] -= b_learning_rate * (b_first_moment + l2_penalty * self.params['b'])

    def init_cache(self):
        cache = dict()
        cache['dW'] = np.zeros_like(self.filters)
        # cache['db'] = np.zeros_like(self.params['b'])
        return cache

    def momentum(self, beta=0.9):
        if not self.momentum_cache:
            self.momentum_cache = self.init_cache()
        self.momentum_cache['dW'] = beta * self.momentum_cache['dW'] + (1 - beta) * self.grad_filters
        # self.momentum_cache['db'] = beta * self.momentum_cache['db'] + (1 - beta) * self.grads['db']

    def rmsprop(self, beta=0.999, amsprop=False):
        if not self.rmsprop_cache:
            self.rmsprop_cache = self.init_cache()

        new_dw = beta * self.rmsprop_cache['dW'] + (1 - beta) * (self.grad_filters**2)
        # new_db = beta * self.rmsprop_cache['db'] + (1 - beta) * (self.grads['db']**2)

        if amsprop:
            self.rmsprop_cache['dW'] = np.maximum(self.rmsprop_cache['dW'], new_dw)
            # self.rmsprop_cache['db'] = np.maximum(self.rmsprop_cache['db'], new_db)
        else:
            self.rmsprop_cache['dW'] = new_dw
            # self.rmsprop_cache['db'] = new_db


if __name__ == '__main__':
    obj = Convolution(num_filters=2)
    print(list(obj.iterate_regions(image=np.array([[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]))))
    obj.forward(input_val=np.array([[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]))
