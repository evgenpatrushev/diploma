import mnist
import numpy as np
from conv import Convolution
from maxpool import MaxPool
from softmax import Softmax

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

l1_conv_start = Convolution(3, image=True)  # 28x28x1 -> 26x26x3
l2_pool = MaxPool()  # 26x26x8 -> 13x13x8
l3_conv = Convolution(5)  # 28x28x1 -> 26x26x3
l4_conv = Convolution(5, shape=2)  # 28x28x1 -> 26x26x3
l5_pool = MaxPool()  # 26x26x8 -> 13x13x8
softmax = Softmax(5 * 5 * 75, 10)  # 13x13x8 -> 10
lr = 0.005

for img, label in zip(train_images, train_labels):
    l1_conv_start.forward(img)
    l2_pool.forward(l1_conv_start.output)
    l3_conv.forward(l2_pool.output)
    l4_conv.forward(l3_conv.output)
    l5_pool.forward(l4_conv.output)
    softmax.forward(l5_pool.output)

    d_l_d_out = -1 * label / softmax.out + (1 - label) / (1 - softmax.out)
    d_l_d_out = softmax.backprop(d_l_d_out, lr)
    d_l_d_out = l5_pool.backprop(d_l_d_out)
    d_l_d_out = l4_conv.back_propagation(d_l_d_out, lr)
    d_l_d_out = l2_pool.backprop(d_l_d_out)
    d_l_d_out = l3_conv.back_propagation(d_l_d_out, lr)
    l1_conv_start.back_propagation(d_l_d_out, lr)


class CNN():
    def __init__(self):
        pass

    def fit(self, image, label):
        out = conv.forward((image / 255) - 0.5)
        out = pool.forward(out)
        out = softmax.forward(out)

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = -1 / (label * out) + (1 - label) / (1 - out)

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc


def run_old_cnn():
    print('MNIST CNN initialized!')

    # Train the CNN for 3 epochs
    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))

        # Shuffle the training data
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        # Train!
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0

            l, acc = train(im, label)
            loss += l
            num_correct += acc

    # Test the CNN
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_images)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)
