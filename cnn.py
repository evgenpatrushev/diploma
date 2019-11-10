import mnist
import numpy as np
from conv import Convolution
from maxpool import MaxPool
from softmax import Softmax
from ReLu import ReLU

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
lr = 0.005
load_weights = False

l1_conv_start = Convolution(num_filters=8, shape=3, image=True)  # 28x28x1 -> 26x26x8
l2_pool = MaxPool(size_of_pool=2)  # 26x26x8 -> 13x13x8
l3_conv = Convolution(num_filters=8, shape=4)  # 13x13x8-> 10x10x64
l4_pool = MaxPool(size_of_pool=2)  # 10x10x64 -> 5x5x64
l5_conv = Convolution(num_filters=5, shape=3)  # 5x5x64 -> 3x3x320
l6_pool = MaxPool(size_of_pool=3)  # 3x3x320 -> 1x1x320
softmax = Softmax(1 * 1 * 320, 10)  # 13x13x8 -> 10


def train(train_x, train_y, p=True):

    if p:
        print('train with permutation')
    else:
        print('train without permutation')
    for epoch in range(4):
        print('--- Epoch %d ---' % (epoch + 1))

        # Shuffle the training data
        if p:
            permutation = np.random.permutation(len(train_x))
            train_img = train_x[permutation]
            train_l = train_y[permutation]
        else:
            train_img = train_x
            train_l = train_y

        loss, num_correct = 0, 0

        for i, obj in enumerate(zip(train_img, train_l)):
            img, l = obj

            l1_conv_start.forward(img)
            l2_pool.forward(l1_conv_start.output)
            l3_conv.forward(l2_pool.output)
            l4_pool.forward(l3_conv.output)
            l5_conv.forward(l4_pool.output)
            l6_pool.forward(l5_conv.output)
            softmax.forward(l6_pool.output)

            label = np.zeros(10)
            label[l] = 1

            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0

            loss += -np.log(softmax.out[l])
            num_correct += 1 if np.argmax(softmax.out) == l else 0

            d_l_d_out = -1 * label / softmax.out

            d_l_d_out = softmax.backprop(d_l_d_out, lr)
            d_l_d_out = l6_pool.backprop(d_l_d_out)
            d_l_d_out = l5_conv.back_propagation(d_l_d_out, lr)
            d_l_d_out = l4_pool.backprop(d_l_d_out)
            d_l_d_out = l3_conv.back_propagation(d_l_d_out, lr)
            d_l_d_out = l2_pool.backprop(d_l_d_out)
            l1_conv_start.back_propagation(d_l_d_out, lr)

    # Test the CNN
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for img, label in zip(test_images, test_labels):
        img = (img / 255) - 0.5

        l1_conv_start.forward(img)
        l2_pool.forward(l1_conv_start.output)
        l3_conv.forward(l2_pool.output)
        l4_pool.forward(l3_conv.output)
        l5_conv.forward(l4_pool.output)
        l6_pool.forward(l5_conv.output)
        softmax.forward(l6_pool.output)

        loss += -np.log(softmax.out[label])
        num_correct += 1 if np.argmax(softmax.out) == label else 0

    num_tests = len(test_images)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)


class CNN():
    def __init__(self):
        pass

    def fit(self, image, label):
        pass


if __name__ == '__main__':
    train(train_images, train_labels)
