import mnist
import numpy as np
from conv import Convolution
from maxpool import MaxPool
from softmax import Softmax
# from ReLu import ReLU

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
lr = 10 ** (-6)
load_weights = False


model = {
    'l1_conv_start': Convolution(num_filters=4, shape=5, image=True),  # 28x28x1 -> 24x24x4
    'l2_pool': MaxPool(size_of_pool=2),  # 24x24x4 -> 12x12x4
    # 'l3_conv': Convolution(num_filters=1, shape=3),  # 12x12x8-> 10x10x8
    # 'l4_pool': MaxPool(size_of_pool=2),  # 10x10x8 -> 5x5x8
    'softmax': Softmax(12 * 12 * 4, 10)}  # 12x12x8 -> 10


# model = {
#     'l1_conv_start': Convolution(num_filters=4, shape=3, image=True),  # 28x28x1 -> 26x26x4
#     'l2_pool': MaxPool(size_of_pool=2),  # 26x26x4 -> 13x13x4
#     'l3_conv': Convolution(num_filters=3, shape=2),  # 13x13x4-> 12x12x12
#     'l4_pool': MaxPool(size_of_pool=2),  # 12x12x12 -> 6x6x12
#     'l5_conv': Convolution(num_filters=3, shape=3),  # 6x6x12-> 4x4x36
#     'l6_pool': MaxPool(size_of_pool=4),  # 4x4x36 -> 1x1x36
#     'softmax': Softmax(1 * 1 * 36, 10)}  # 5x5x64 -> 10


def train(train_x, train_y, p=False):
    if p:
        print('train with permutation')
    else:
        print('train without permutation')
    for epoch in range(6):
        print('--- Epoch %d ---' % (epoch + 1))

        # Shuffle the training data
        if p:
            permutation = np.random.permutation(len(train_x))
            train_img = train_x[permutation]
            train_l = train_y[permutation]
        else:
            train_img = train_x
            train_l = train_y

        loss, num_correct, iter = 0, 0, 1

        for i, obj in enumerate(zip(train_img, train_l)):
            output, label_num = obj

            for name in model:
                model[name].forward(output)
                output = model[name].output

            label = np.zeros(10)
            label[label_num] = 1

            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0
                iter += i

            loss += -np.log(model['softmax'].output[label_num])
            num_correct += 1 if np.argmax(model['softmax'].output) == label_num else 0

            d_l_d_out = -1 * label / model['softmax'].output

            for layer in reversed(list(model.values())):
                d_l_d_out = layer.backprop(d_l_d_out)
            #     if "first_momentum" in dir(layer) and "second_momentum" in dir(layer):
            #         layer.first_momentum()
            #         layer.second_momentum()

            for layer in reversed(list(model.values())):
                if "update_weights" in dir(layer):
                    layer.update_weights(correct_bias=True, iter=iter, optimization='not adam')


# Test the CNN
    # print('\n--- Testing the CNN ---')
    # loss = 0
    # num_correct = 0
    # for img, label in zip(test_images, test_labels):
    #     img = (img / 255) - 0.5
    #
    #     l1_conv_start.forward(img)
    #     l2_pool.forward(l1_conv_start.output)
    #     l3_conv.forward(l2_pool.output)
    #     l4_pool.forward(l3_conv.output)
    #     softmax.forward(l4_pool.output)
    #
    #     loss += -np.log(softmax.out[label])
    #     num_correct += 1 if np.argmax(softmax.out) == label else 0
    #
    # num_tests = len(test_images)
    # print('Test Loss:', loss / num_tests)
    # print('Test Accuracy:', num_correct / num_tests)


class CNN:
    def __init__(self):
        pass

    def fit(self, image, label):
        pass


if __name__ == '__main__':
    train(train_images, train_labels)
