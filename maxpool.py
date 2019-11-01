import numpy as np
import warnings


class MaxPool:
    # A Max Pooling layer using a pool size of 2.
    def __init__(self, size_of_pool=2):
        self.size_of_pool = size_of_pool
        self.input, self.output = np.array([]), np.array([])

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''

        _, h, w = image.shape

        new_h = h // self.size_of_pool
        new_w = w // self.size_of_pool

        regions = ((i, j, image[(i * self.size_of_pool):(i * self.size_of_pool + self.size_of_pool),
                          (j * self.size_of_pool):(j * self.size_of_pool + self.size_of_pool)])
                   for i in range(new_h)
                   for j in range(new_w))

        for i, j, region in regions:
            yield i, j, region

    def forward(self, input_val):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''

        self.input = input_val

        h, w, _ = input_val.shape

        if h % self.size_of_pool != 0 or w % self.size_of_pool != 0:
            warnings.warn("Warning, pict don't cross all pool operation")

        self.output = np.zeros((h // self.size_of_pool, w // self.size_of_pool, _))

        for i, j, im_region in self.iterate_regions(input_val):
            self.output[i, j] = np.amax(im_region, axis=(0, 1))

        return self.output

    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        d_L_d_input = np.zeros(self.input.shape)

        for i, j, im_region in self.iterate_regions(self.input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.size_of_pool + i2, j * self.size_of_pool + j2, f2] = d_L_d_out[
                                i, j, f2]

        return d_L_d_input


if __name__ == '__main__':
    from conv import Convolution
    obj = Convolution(num_filters=2, image=True)
    print(list(obj.iterate_regions(image=np.array([[0, 50, 0, 29],
                                                   [0, 80, 31, 2],
                                                   [33, 90, 0, 75],
                                                   [0, 9, 0, 95]]))))
    obj.forward(input_val=np.array([[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]))
    obj_max = MaxPool()
    # print(list(obj.iterate_regions(image=np.array([[[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]]))))
    # print(obj.forward(input=np.array([[[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]]])))
    obj_max.forward(input_val=obj.output)
    obj_max.backprop(d_L_d_out=np.random.random(obj_max.output.shape))
