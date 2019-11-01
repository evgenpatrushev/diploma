import numpy as np


class Convolution:
    # A Convolution layer using (shape x shape) filters.

    def __init__(self, num_filters=1, shape=3, stride=1, image=False):
        self.image = image
        self.num_filters = num_filters
        self.stride = stride
        self.shape = shape
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, shape, shape) / 9
        self.biases = np.ones(num_filters)
        self.input, self.output = np.array([]), np.array([])

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
                num_filters, shape = self.filters.shape
                filters = np.random.randn(3, num_filters, shape, shape) / 9
                output = 0

                for img_color, filter_for_color in zip(input_val, filters):
                    self.filters = filter_for_color
                    self._forward(input_val=img_color)
                    output += self.output

                self.filters = filters
                self.output = output + self.biases

        else:
            num_conv = input_val.shape[2]
            output = 0
            for i in range(num_conv):
                self._forward(input_val=input_val[:, :, i])
                self.output += self.biases
                if output:
                    output = self.output
                else:
                    output = np.concatenate((output, self.output), axis=2)

        self.input = input_val

    def _forward(self, input_val):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''

        h, w = input_val.shape
        self.output = np.zeros(
            ((h - self.shape) // self.stride + 1, (w - self.shape) // self.stride + 1, self.num_filters))

        for i, j, im_region in self.iterate_regions(input_val):
            self.output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return self.output

    def back_propagation(self, d_L_d_out, learn_rate):

        if self.image:

            if len(self.input.shape) == 2:
                self._backprop(d_L_d_out, learn_rate, self.input)

            elif len(self.input.shape) == 3:
                filters = self.filters
                for i, img_color, filters_for_color in enumerate(zip(self.input, filters)):
                    self.filters = filters_for_color
                    self._backprop(d_L_d_out=d_L_d_out, learn_rate=learn_rate, input_val=img_color)
                    filters[i] = self.filters

        else:
            """
            d_L_d_out have to be matrix same size as output of forward function 
            but with error for backprop
            """
            num_conv = self.input.shape[2]

            for i in range(num_conv):
                self._backprop(d_L_d_out=d_L_d_out[:, :, i * self.shape:i + self.shape], learn_rate=learn_rate,
                               input_val=self.input[:, :, i])
                # TODO -sure about backprop at multiple conv obj with same filters (formula) ... look good but not sure

    def _backprop(self, d_L_d_out, learn_rate, input_val):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.

        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for i, j, im_region in self.iterate_regions(input_val):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # TODO -add biases update

        return None


if __name__ == '__main__':
    obj = Convolution(num_filters=2)
    print(list(obj.iterate_regions(image=np.array([[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]))))
    o = obj.forward(input_val=np.array([[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]))
    print(o)