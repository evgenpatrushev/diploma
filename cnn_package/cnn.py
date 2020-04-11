import mnist
import numpy as np
from cnn_package.conv import Convolution
from cnn_package.maxpool import MaxPool
from cnn_package.softmax import Softmax
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
    'softmax': Softmax(12 * 12 * 4, 10)}  # 12x12x8 -> 10


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


if __name__ == '__main__':
    train(train_images, train_labels)
