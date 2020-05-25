import os
import csv
import fnmatch

import warnings
from typing import Dict

warnings.filterwarnings('ignore')

import time as t
import numpy as np
import pandas as pd
from isic_api import ISICApi
from threading import Thread

import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from cnn_package.conv import Convolution
from cnn_package.maxpool import MaxPool
from cnn_package.softmax import Softmax

# coding=utf-8
import keras.backend as k
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


def find(pattern, path=''):
    result = []
    if not path:
        path = os.curdir
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


class DownloadThreadImgInfo(Thread):
    """
    Class obj for downloading img info by multi threads
    """

    def __init__(self, count, from_count, name):
        Thread.__init__(self)
        self.count = count
        self.from_count = from_count
        self.thread_name = name

    def run(self):
        print(f"THREAD INFO: {self.thread_name} start working: from_count = {self.from_count}, count = {self.count}")

        data = DownloadDataset()
        data.download_info(count=self.count, from_count=self.from_count, thread_name=self.thread_name,
                           just_melanoma=True)

        print(f"THREAD INFO: {self.thread_name} end working ")


class DownloadThreadImg(Thread):
    """
    Class obj for downloading img by multi threads
    """

    def __init__(self, data_frame, thread_name):
        Thread.__init__(self)
        self.data_frame = data_frame
        self.thread_name = thread_name

    def run(self):
        print(f"THREAD INFO: {self.thread_name} start working: count_img = {self.data_frame.shape[0]}")

        data_handler = DownloadDataset()
        data_handler.download_img(thread_name=self.thread_name, df=self.data_frame)

        print(f"THREAD INFO: {self.thread_name} end working ")


class DownloadDataset:
    """
    Class for handling data manipulation
    """

    def __init__(self, create_connection=True, login_file='data_login.txt', path_archive='ISICArchive'):

        self.path_archive = path_archive
        self.save_path_img = os.path.join(path_archive, 'images')

        self.login_file = os.path.join(self.path_archive, login_file)
        self.file_name_metadata = os.path.join(self.path_archive, 'imagedata.csv')
        self.train_file = os.path.join(self.path_archive, 'Train.csv')
        self.test_file = os.path.join(self.path_archive, 'Test.csv')

        if create_connection:
            self.api_connection = self.create_connection_api()
        else:
            self.api_connection = None

    def create_connection_api(self):
        """
        Creating api connection with ISIC
        :return: connection
        """

        with open(self.login_file) as f:
            name, password = f.readline().split(',')
        return ISICApi(username=name, password=password)

    def download_info_threading(self, list_params=None):
        """
        Downloading img info by multi threads

        :param list_params: list with two elements for element: [[count_1, from_1], ...],
        where count_1 - count img, from_1 - from what img to start for first threat

        :return: None
        """
        if list_params is None:
            list_params = [[1000, 0],
                           [1000, 1000],
                           [1000, 2000],
                           [1000, 3000],
                           [1000, 4000],
                           [1000, 5000],
                           [1000, 6000],
                           [1000, 7000],
                           [1000, 8000],
                           [1000, 9000],
                           [1000, 10000],
                           [1000, 11000],
                           [1000, 12000],
                           [1000, 13000],
                           [1000, 14000],
                           [1000, 15000],
                           [1000, 16000],
                           [1000, 17000],
                           [1000, 18000],
                           [1000, 19000],
                           [1000, 20000],
                           [1000, 21000],
                           [1000, 22000],
                           [906, 23000]]

        if not self.api_connection:
            self.create_connection_api()

        for i, params in enumerate(list_params):
            name_of_thread = f"thread_{i + 1}"
            thread = DownloadThreadImgInfo(name=name_of_thread, count=params[0], from_count=params[1])
            thread.start()
            t.sleep(5)

    def download_info(self, count=3000, from_count=0, append=False, thread_name='', just_melanoma=False):
        """
        Downloading img info (can used be by multi thread)

        :param count: count img (int)
        :param from_count: from what number to start (int)
        :param append: append into file_name_metadata or not (bool)
        :param thread_name: thread name for logs (str)
        :param just_melanoma: down only melanoma or not (bool)
        :return:
        """
        if not os.path.isfile(self.file_name_metadata):

            if not self.api_connection:
                self.create_connection_api()

            images = self.api_connection.getJson(f'image?limit={count + from_count}&offset=0&sort=name')[from_count:]
            print(f'{f"{thread_name}:  " if thread_name else thread_name}Fetching metadata for {len(images)} images')
            print(f'{f"{thread_name}:  " if thread_name else thread_name}Start name: {images[0]["name"]} end '
                  f'{images[-1]["name"]}')
            image_details = []
            start = t.time()
            iteration = 0
            try:
                for img in images:
                    # Fetch the full image details
                    image_detail = self.api_connection.getJson(f'image/{img["_id"]}')

                    if just_melanoma and ('melanocytic' in image_detail['meta']['clinical']) and \
                            image_detail['meta']['clinical']['melanocytic'] and \
                            image_detail['meta']['clinical']['benign_malignant'] == 'malignant':
                        print(f'{f"{thread_name}:  " if thread_name else thread_name}append malignant img')
                        image_details.append(image_detail)
                    elif not just_melanoma:
                        image_details.append(image_detail)

                    iteration += 1
                    if iteration % 99 == 0:
                        time_to_end = np.round((count - iteration) * np.round(t.time() - start, 2) / 100, 2)
                        print(
                            f'\n {f"{thread_name}:  " if thread_name else thread_name}'
                            f'download 100 obj; '
                            f'time went sec = {np.round(t.time() - start, 2)}, prediction time for end = '
                            f'{time_to_end} sec or {np.round(time_to_end / 60, 2)} min')
                        start = t.time()
            except Exception as e:
                print(
                    f'{f"{thread_name}:  " if thread_name else thread_name}uuuups, problem at downloading img info {e}')
            finally:
                print(f'{f"{thread_name}:  " if thread_name else thread_name}get meta fields')
                metadata_fields = set(
                    field
                    for imageDetail in image_details
                    for field in imageDetail['meta']['clinical'].keys()
                )
                metadata_fields = ['isic_id'] + sorted(metadata_fields)

                # Write the metadata to a CSV
                csv_name = self.file_name_metadata + f'{f"_{thread_name}" if thread_name else thread_name}' + '.csv'
                print(f'{f"{thread_name}:  " if thread_name else thread_name}'
                      f'Writing metadata to CSV: {csv_name}')

                if append:
                    type_file = 'a'
                else:
                    type_file = 'w'

                with open(csv_name, type_file) as file:

                    csv_w = csv.DictWriter(file, metadata_fields)
                    csv_w.writeheader()

                    for imageDetail in image_details:
                        row_dict = imageDetail['meta']['clinical'].copy()
                        row_dict['isic_id'] = imageDetail['name']
                        csv_w.writerow(row_dict)

    def img_down_thread_creating(self, list_params=None, count_in_thread=100):
        """
        Downloading img by multi threads

        :param list_params: list of data frames: [df_1, df_2, ...]
        :param count_in_thread: count to split
        :return:
        """

        if list_params is None:
            df = self._create_dataset_frame()
            list_params = np.array_split(df, df.shape[0] // count_in_thread)

        if not self.api_connection:
            self.create_connection_api()

        for i, params in enumerate(list_params):
            name_of_thread = f"thread_{i + 1}"
            thread = DownloadThreadImg(thread_name=name_of_thread, data_frame=params)
            thread.start()
            t.sleep(5)

    def _create_dataset_frame(self):
        """
        Specific function for processing two csv files:
        with metadata of combine benign_melanoma values and just melanoma
        :return: DataFrame
        """

        df_all = pd.read_csv(self.file_name_metadata)

        df_melanoma = pd.read_csv(os.path.join(self.path_archive, 'melanoma_dataset.csv'))
        df_benign = df_all[df_all.benign_malignant == 'benign']

        df_all = df_benign[:1800].append(df_melanoma)
        df_all['name'] = df_all['isic_id']

        del df_all['isic_id']
        return df_all

    def download_img(self, count=0, df=None, thread_name=''):
        """
        Download img

        :param count: count img
        :param df: dataframe with names of img
        :param thread_name: name of thread
        :return:
        """
        c = 10

        if not os.path.exists(self.save_path_img):
            os.makedirs(self.save_path_img)

        if not self.api_connection:
            self.create_connection_api()

        if not count:
            def get_id(name_c):
                return self.api_connection.getJson(f'image?name={name_c}')[0]['_id']

            df['id'] = df['name'].apply(get_id)
            image_list = [{'name': n, '_id': _id} for n, _id in zip(df['name'].values, df['id'].values)]
        else:
            image_list = self.api_connection.getJson(f'image?limit={count}&offset=0&sort=name')

        count_img = len(image_list)
        start = t.time()
        print(f'{f"{thread_name}:  " if thread_name else thread_name}Downloading {count_img} images')
        for i, img in enumerate(image_list):

            if i % (c - 1) == 0:
                time_to_end = np.round((count_img - i) * np.round(t.time() - start, 2) / c, 2)
                print(
                    f'\n{f"{thread_name}:  " if thread_name else thread_name}'
                    f'download {c} obj; time went sec = {np.round(t.time() - start, 2)}, prediction time for end = '
                    f'{time_to_end} sec or {np.round(time_to_end / 60, 2)} min')
                start = t.time()

            if find(f'{img["name"]}.jpg', path=self.save_path_img):
                continue

            image_file_output_path = os.path.join(self.save_path_img, f'{img["name"]}.jpg')
            image_file = self.api_connection.get(f'image/{img["_id"]}/download')
            image_file.raise_for_status()
            with open(image_file_output_path, 'wb') as imageFileOutputStream:
                for chunk in image_file:
                    imageFileOutputStream.write(chunk)

    def load_data(self, img_rows, img_cols, gray=False, sharped=False, normal=True, plot=True, save_size=False):
        """
        Loading dataset

        :param save_size:
        :param img_rows: count of rows (int)
        :param img_cols: count of columns (int)
        :param gray: download gray or RGB (bool)
        :param sharped: create sharped img (bool)
        :param normal: normalize img (bool)
        :param plot: plotting statistic for all(bool)
        :return: test and train data frames
        """
        return_dict = {}

        x_train = pd.read_csv(self.train_file)
        y_train = x_train['benign_malignant'].values

        x_test = pd.read_csv(self.test_file)
        y_test = x_test['benign_malignant'].values

        if plot:
            #  Stat plot of data:
            df_stat = pd.DataFrame([[y_train[y_train == 0].size, y_test[y_test == 0].size,
                                     y_train[y_train == 0].size + y_test[y_test == 0].size],
                                    [y_train[y_train == 1].size, y_test[y_test == 1].size,
                                     y_train[y_train == 1].size + y_test[y_test == 1].size]],
                                   columns=['count of train', 'count of test', 'count of test + train'],
                                   index=['benign', 'malignant'])
            ax = df_stat.plot.barh(rot=0, subplots=True)
            ax[0].set_title('train observe objects')
            ax[1].set_title('test observe objects')
            ax[2].set_title('all observe objects')
            plt.show()

        download = False

        if save_size:
            img_path = os.path.join(self.path_archive, f'images_{img_rows}x{img_rows}')
            if not os.path.exists(img_path):
                os.mkdir(img_path)
                img_path = self.save_path_img
                download = True
        else:
            img_path = self.save_path_img

        #  Download img:
        images_train = []
        for index, row in x_train.iterrows():
            if gray:
                images_train.append(image.img_to_array(
                    image.load_img(os.path.join(img_path, f"{row['name']}.jpg"),
                                   target_size=(img_rows, img_cols), color_mode="grayscale"))[:, :, 0])
            else:
                images_train.append(image.img_to_array(
                    image.load_img(os.path.join(img_path, f"{row['name']}.jpg"),
                                   target_size=(img_rows, img_cols))))
            if download:
                image.save_img(os.path.join(self.path_archive, f'images_{img_rows}x{img_rows}', f"{row['name']}.jpg"),
                               images_train[-1])

        images_train = np.array(images_train)

        images_test = []
        for index, row in x_test.iterrows():
            if gray:
                images_test.append(image.img_to_array(
                    image.load_img(os.path.join(img_path, f"{row['name']}.jpg"),
                                   target_size=(img_rows, img_cols), color_mode="grayscale"))[:, :, 0])
            else:
                images_test.append(image.img_to_array(
                    image.load_img(os.path.join(img_path, f"{row['name']}.jpg"),
                                   target_size=(img_rows, img_cols))))
            if download:
                image.save_img(os.path.join(self.path_archive, f'images_{img_rows}x{img_rows}', f"{row['name']}.jpg"),
                               images_test[-1])

        images_test = np.array(images_test)

        images_train = images_train.astype('float32')
        images_test = images_test.astype('float32')

        return_dict['original'] = (images_train, images_test)

        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17, 8))
        ax1.hist(images_train.flatten(), color='r', label='train', alpha=0.8)
        ax1.hist(images_test.flatten(), color='g', label='test', alpha=0.8)
        ax1.legend(loc=4)
        ax1.set_title(f'Hist of images pixel values (not normalized) size {img_rows}x{img_cols}')

        n_train, bins_train, _ = ax2.hist(images_train.flatten(), density=True, cumulative=True, color='b',
                                          label='train general', alpha=0.8, bins=25)
        n_test, bins_test, _ = ax2.hist(images_test.flatten(), density=True, cumulative=True, color='y',
                                        label='test general', alpha=0.3, bins=25)

        ax2.legend(loc=4)
        ax2.set_title(f'Hist of images pixel values (cumulative and normalized) size {img_rows}x{img_cols}')

        if sharped:
            images_train_shr = np.interp(images_train.flatten(), bins_train[:-1], n_train * 255). \
                reshape(images_train.shape)
            images_test_shr = np.interp(images_test.flatten(), bins_test[:-1], n_test * 255). \
                reshape(images_test.shape)

            ax2.hist(images_train_shr.flatten(), cumulative=True, density=True, color='b',
                     label='train and test sharped', fill=False, bins=25)
            ax2.legend(loc=4)
            return_dict['sharped'] = (images_train_shr, images_test_shr)

            if normal:
                #  Normalization of data
                images_train_shr /= 255
                images_test_shr /= 255
                return_dict['sharped_normal'] = (images_train_shr, images_test_shr)

        plt.show()

        if normal:
            #  Normalization of data
            images_train /= 255
            images_test /= 255
            return_dict['normal'] = (images_train, images_test)

        y_train = np_utils.to_categorical(y_train, 2)
        y_test = np_utils.to_categorical(y_test, 2)

        return return_dict, y_train, y_test


class HandlerCNN:
    models: Dict[int, Sequential]

    def __init__(self, rows=34, cols=34, gray=False, path_to_archive='ISICArchive', models_dir='cnn_package/models'):

        self.decay_const = 1e-6
        self.momentum_const = 0.9
        self.validation_split_const = 0.1
        self.learning_rate_const = 0.001
        self.batch_size_const = 15
        self.nb_epoch_const = 10
        self.steps_per_epoch_const = 500

        self.early_stop_param = 30
        self.early_stop = True

        def scheduler(epoch):
            epoch_count = 50
            coefficient = 1

            if epoch < epoch_count:
                return self.learning_rate_const / coefficient
            else:
                return self.learning_rate_const * tf.math.exp(0.07 * (epoch_count - epoch)) / coefficient

        self.scheduler_function = scheduler
        self.schedule_bool = False

        self.path_to_archive = path_to_archive
        self.models_down_dir = models_dir

        self.rows, self.cols = 0, 0

        if gray:
            self.gray = 0
        else:
            self.gray = 3

        self.dataset, self.y_train, self.y_test = np.array([]), np.array([]), np.array([])

        self.model1, self.model2, self.model3, self.model4 = None, None, None, None

        print(f'Loading dataset for {rows} rows and {cols} col')
        self.load_data(row=rows, col=cols, plot=True, gray=gray)
        print('Loading dataset end')

    def load_data(self, row, col, plot=False, gray=False):
        if row == self.rows and col == self.cols and self.gray == gray:
            return 0

        self.rows, self.cols = row, col

        if gray:
            self.gray = 0
        else:
            self.gray = 3
        data_h = DownloadDataset(create_connection=False, login_file='', path_archive=self.path_to_archive)

        self.dataset, self.y_train, self.y_test = data_h.load_data(img_rows=row, img_cols=col, gray=bool(gray),
                                                                   sharped=True, normal=True, plot=plot, save_size=True)
        self.load_models()

    def load_models(self):
        self.model1 = Sequential()
        self.model1.add(
            Conv2D(32, (3, 3), padding='same', input_shape=(self.rows, self.cols, self.gray), activation='relu'))
        self.model1.add(MaxPooling2D(pool_size=(2, 2)))
        self.model1.add(Dropout(0.25))

        self.model1.add(Flatten())
        self.model1.add(Dense(2, activation='softmax'))

        self.model2 = Sequential()
        self.model2.add(
            Conv2D(32, (3, 3), padding='same', input_shape=(self.rows, self.cols, self.gray), activation='relu'))
        self.model2.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model2.add(MaxPooling2D(pool_size=(2, 2)))
        self.model2.add(Dropout(0.25))

        self.model2.add(Flatten())
        self.model2.add(Dense(2, activation='softmax'))

        self.model3 = Sequential()
        self.model3.add(
            Conv2D(32, (3, 3), padding='same', input_shape=(self.rows, self.cols, int(self.gray)), activation='relu'))
        self.model3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model3.add(MaxPooling2D(pool_size=(2, 2)))
        self.model3.add(Dropout(0.25))

        self.model3.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model3.add(Conv2D(32, (3, 3), activation='relu'))
        self.model3.add(MaxPooling2D(pool_size=(2, 2)))
        self.model3.add(Dropout(0.25))

        self.model3.add(Flatten())
        self.model3.add(Dense(2, activation='softmax'))

        self.model4 = Sequential()
        self.model4.add(
            Conv2D(32, (3, 3), padding='same', input_shape=(self.rows, self.cols, int(self.gray)), activation='relu'))
        self.model4.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model4.add(MaxPooling2D(pool_size=(2, 2)))
        self.model4.add(Dropout(0.25))

        self.model4.add(Conv2D(32, (5, 5), padding='same', activation='sigmoid'))
        self.model4.add(Conv2D(32, (5, 5), activation='sigmoid'))
        self.model4.add(MaxPooling2D(pool_size=(2, 2)))
        self.model4.add(Dropout(0.25))

        self.model4.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model4.add(Conv2D(32, (3, 3), activation='relu'))
        self.model4.add(MaxPooling2D(pool_size=(2, 2)))
        self.model4.add(Dropout(0.25))

        self.model4.add(Flatten())
        self.model4.add(Dense(2, activation='softmax'))

        self.models = {1: self.model1, 2: self.model2, 3: self.model3, 4: self.model4}

    @staticmethod
    def reset_weights(model):
        session = k.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)

    def load_cnn(self, x_train, x_test, y_train, y_test, model, optimizer, nb_epoch=None, reset_weights_bool=False,
                 batch_size=None, model_name='', plotting=True, generator=True, steps_per_epoch=None, logging=2,
                 measurement=True):

        if nb_epoch is None:
            nb_epoch = self.nb_epoch_const
        if batch_size is None:
            batch_size = self.batch_size_const
        if steps_per_epoch is None:
            steps_per_epoch = self.steps_per_epoch_const
        if reset_weights_bool:
            self.reset_weights(model)

        callbacks = []
        if self.early_stop:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.early_stop_param))
        if self.schedule_bool:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.scheduler_function))
        if not callbacks:
            callbacks = None

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        if generator:
            # construct the training image generator for data augmentation
            aug = image.ImageDataGenerator(rotation_range=40,
                                           shear_range=0.15,
                                           zoom_range=0.15,
                                           horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
            # train the network
            # todo add validation split
            fitted_model = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                               validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch,
                                               epochs=nb_epoch, verbose=logging, use_multiprocessing=False,
                                               callbacks=callbacks)
        else:
            fitted_model = model.fit(x_train, y_train,
                                     batch_size=batch_size,
                                     epochs=nb_epoch,
                                     validation_split=self.validation_split_const,
                                     shuffle=True,
                                     verbose=logging,
                                     use_multiprocessing=False,
                                     callbacks=callbacks)

        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy on test data: %.2f%%" % (scores[1] * 100))

        if model_name:
            model.save(os.path.join(self.models_down_dir, f'{model_name}.h5'))

        if plotting:
            plt.style.use("ggplot")
            f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 7))
            ax1.plot(fitted_model.epoch, fitted_model.history["loss"], label="train_loss")
            ax1.plot(fitted_model.epoch, fitted_model.history["val_loss"], label="val_loss")
            ax2.plot(fitted_model.epoch, fitted_model.history["accuracy"], label="train_accuracy")
            ax2.plot(fitted_model.epoch, fitted_model.history["val_accuracy"], label="val_accuracy")
            ax1.set_title(
                f"Training Loss and Accuracy on Dataset \n{f'model name = {model_name}' if model_name else ''}")
            ax1.set_ylabel('Loss')
            ax2.set_ylabel('Accuracy')
            plt.xlabel("Epoch #")
            ax1.legend(loc="lower left")
            ax2.legend(loc="lower left")
            plt.show()

        if measurement:
            self.metrics_cnn(model.predict(x_test))

        return fitted_model, np.round(scores[1] * 100, 2)

    @staticmethod
    def my_cnn(images, images_y, epo, model, lr=0.001):
        for epoch in range(epo):
            print('--- Epoch %d ---' % (epoch + 1))

            # shuffle the training data
            permutation = np.random.permutation(len(images))
            train_img = images[permutation]
            train_l = images_y[permutation]

            loss, num_correct, iterations = 0, 0, 1

            # training ...
            for i, obj in enumerate(zip(train_img, train_l)):

                # data get
                output, label = obj
                label_num = np.argmax(label)
                output = np.moveaxis(output, -1, 0)

                # forward going
                for name in model:
                    model[name].forward(output)
                    output = model[name].output

                # logging into
                if i % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, loss / 100, num_correct)
                    )
                    loss = 0
                    num_correct = 0
                    iterations += i

                # calculate backprop values
                loss += -np.log(model['softmax'].output[label_num])
                num_correct += 1 if np.argmax(model['softmax'].output) == label_num else 0

                d_l_d_out = -1 * label / model['softmax'].output

                for layer in reversed(list(model.values())):
                    d_l_d_out = layer.backprop(d_l_d_out)

                # update values
                for layer in reversed(list(model.values())):
                    if "update_weights" in dir(layer):
                        layer.update_weights(correct_bias=True, iter=iterations, optimization='not adam',
                                             learning_rate=lr)

        def predict(images_pr):
            result = []
            for img in images_pr:
                out = img
                for layer_name in model:
                    model[layer_name].forward(out)
                    out = model[layer_name].output
                result.append(np.argmax(out))
            return result

        return predict

    def cnn_diff_models(self, row=34, col=34, count_epoch=None, optimum_method_name='sgd', reset_weight=True,
                        list_of_models=None, norm=True, sharp=False, load_flag=False, generator=False):
        """
        Create different models of keras CNN and fit it on data

        :param reset_weight:
        :param generator:
        :param load_flag:
        :param count_epoch:
        :param sharp: sharped data, bool
        :param norm: normalization of data, bool
        :param col: columns count, int
        :param row: rows count, int
        :param optimum_method_name: name of optimize method
        :param list_of_models: list of models to use, list/set
        :return: fitted models and their accuracy on test data
        """

        if count_epoch is None:
            count_epoch = self.nb_epoch_const
        if list_of_models is None:
            list_of_models = {1, 2, 3, 4}

        if sharp and norm:
            train, test = self.dataset['sharped_normal']
        elif sharp:
            train, test = self.dataset['sharped']
        elif norm:
            train, test = self.dataset['normal']
        else:
            train, test = self.dataset['original']

        result = []

        if optimum_method_name.lower() == 'sgd':
            optimum_method = SGD(lr=self.learning_rate_const, decay=self.decay_const,
                                 momentum=self.momentum_const, nesterov=True)
        elif optimum_method_name.lower() == 'rmsprop':
            optimum_method = RMSprop(learning_rate=self.learning_rate_const, rho=0.9)
        elif optimum_method_name.lower() == 'adagrad':
            optimum_method = Adagrad(learning_rate=self.learning_rate_const)
        elif optimum_method_name.lower() == 'adadelta':
            optimum_method = Adadelta(learning_rate=self.learning_rate_const, rho=0.9)
        elif optimum_method_name.lower() == 'adam':
            optimum_method = Adam(learning_rate=self.learning_rate_const, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif optimum_method_name.lower() == 'adamax':
            optimum_method = Adamax(learning_rate=self.learning_rate_const, beta_1=0.9, beta_2=0.999)
        elif optimum_method_name.lower() == 'nadam':
            optimum_method = Nadam(learning_rate=self.learning_rate_const, beta_1=0.9, beta_2=0.999)
        else:
            return 0

        for i in list_of_models:
            if i in self.models:
                # -------------------
                print(f'\n\nCreating CNN keras model #{i}')
                name = f'model_{i}_{optimum_method_name.lower()}_{row}x{col}'
                if load_flag and find(f'{name}.h5', path=self.models_down_dir):
                    print('Loading model ... ')
                    fitted_model = load_model(f'{name}.h5')
                    accuracy_model = np.round(fitted_model.evaluate(test, self.y_test, verbose=0)[1] * 100, 2)
                    print("Accuracy on test data: %.2f%%" % accuracy_model)
                else:

                    fitted_model, accuracy_model = self.load_cnn(x_train=train, x_test=test, y_train=self.y_train,
                                                                 y_test=self.y_test, reset_weights_bool=reset_weight,
                                                                 model=self.models[i], model_name=name,
                                                                 optimizer=optimum_method,
                                                                 nb_epoch=count_epoch, generator=generator)
                result.append([fitted_model, accuracy_model])

        return result

    def batch_grid(self, model, x_train, y_train, x_test, y_test, nb_epoch, grid_range=None, optimizer=None, axis=None,
                   title='grid batch', color='r'):
        """
        Create grid for range of batch size and create a plot of it with specific optimization method

        :param optimizer:
        :param model: keras CNN model with created layers
        :param x_train: images for training
        :param y_train: labels for training
        :param x_test: images for training
        :param y_test: labels for training
        :param nb_epoch: number of epoch
        :param grid_range: range of batch size
        :param axis: ax of plot
        :param title: label of plot
        :param color: color of plot
        :return: DataFrame of values accuracy and batch size
        """

        if grid_range is None:
            grid_range = [1] + [i for i in range(5, 61, 5)]

        if optimizer is None:
            optimizer = SGD(lr=self.learning_rate_const, decay=self.decay_const, momentum=self.momentum_const,
                            nesterov=True)

        df = pd.DataFrame(columns=['accuracy', 'batch size'])
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        for batch_size in grid_range:
            model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                      validation_split=self.validation_split_const, shuffle=True, verbose=2)

            scores = model.evaluate(x_test, y_test, verbose=0)
            df = df.append({'accuracy': np.round(scores[1] * 100, 2), 'batch size': batch_size}, ignore_index=True)

        df = df.astype({'batch size': int})
        print(df)
        print('-------------------------------------------------------------------------------------------------------')
        df.plot(x='batch size', y='accuracy', label=title, ax=axis, color=color, style='.-')
        return df

    def learning_rate_grid(self, model, x_train, y_train, x_test, y_test, nb_epoch, batch_s, grid_range=None,
                           axis=None, title='grid learning rate', color='r'):
        """
        Create grid for range of learning rate and create a plot of it with specific optimization method

        :param batch_s: batch size
        :param model: keras CNN model with created layers
        :param x_train: images for training
        :param y_train: labels for training
        :param x_test: images for training
        :param y_test: labels for training
        :param nb_epoch: number of epoch
        :param grid_range: range of learning range
        :param axis: ax of plot
        :param title: label of plot
        :param color: color of plot
        :return: DataFrame of values accuracy and learning rate
        """

        if grid_range is None:
            grid_range = [10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 2 * 10 ** (-1), 3 * 10 ** (-1),
                          4 * 10 ** (-1), 5 * 10 ** (-1), 6 * 10 ** (-1), 7 * 10 ** (-1), 8 * 10 ** (-1),
                          9 * 10 ** (-1),
                          10 ** 0]

        df = pd.DataFrame(columns=['accuracy', 'learning rate'])

        for learning_rate in grid_range:
            sgd = SGD(lr=learning_rate, decay=self.decay_const, momentum=self.momentum_const, nesterov=True)

            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_s, epochs=nb_epoch, validation_split=0.1, shuffle=True,
                      verbose=2)

            scores = model.evaluate(x_test, y_test, verbose=0)
            df = df.append({'accuracy': np.round(scores[1] * 100, 2), 'learning rate': learning_rate},
                           ignore_index=True)

        print(df)
        print('-------------------------------------------------------------------------------------------------------')
        df.plot(x='learning rate', y='accuracy', label=title, ax=axis, color=color, style='.-')
        return df

    def optimizer_grid(self, model, x_train, y_train, x_test, y_test, nb_epoch, batch_s, grid_range=None, axis=None,
                       title='different optimize methods', color='r'):
        """
        Create grid for range of optimizer methods and create a plot of it

        :param batch_s: batch size
        :param model: keras CNN model with created layers
        :param x_train: images for training
        :param y_train: labels for training
        :param x_test: images for training
        :param y_test: labels for training
        :param nb_epoch: number of epoch
        :param grid_range: range of optimizer methods
        :param axis: ax of plot
        :param title: label of plot
        :param color: color of plot
        :return: DataFrame of values accuracy and optimizer methods
        """

        if grid_range is None:
            grid_range = [
                (SGD(lr=self.learning_rate_const, decay=self.decay_const, momentum=self.momentum_const, nesterov=True),
                 'SGD'),
                (RMSprop(learning_rate=self.learning_rate_const, rho=0.9), 'RMSprop'),
                (Adagrad(learning_rate=self.learning_rate_const), 'Adagrad'),
                (Adadelta(learning_rate=self.learning_rate_const, rho=0.9), 'Adadelta'),
                (Adam(learning_rate=self.learning_rate_const, beta_1=0.9, beta_2=0.999, amsgrad=False), 'Adam'),
                (Adamax(learning_rate=self.learning_rate_const, beta_1=0.9, beta_2=0.999), 'Adamax'),
                (Nadam(learning_rate=self.learning_rate_const, beta_1=0.9, beta_2=0.999), 'Nadam')]

        df = pd.DataFrame(columns=['accuracy', 'optimize method'])

        for method, title_of_method in grid_range:
            model.compile(loss='categorical_crossentropy',
                          optimizer=method,
                          metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_s, epochs=nb_epoch, validation_split=0.1, shuffle=True,
                      verbose=2)

            scores = model.evaluate(x_test, y_test, verbose=0)
            df = df.append({'accuracy': np.round(scores[1] * 100, 2), 'optimize method': title_of_method},
                           ignore_index=True)

        print(df)
        print('-------------------------------------------------------------------------------------------------------')
        df.plot(x='optimize method', y='accuracy', label=title, ax=axis, color=color, style='.-')
        return df

    def compare_models(self, number_rows=34, number_cols=34, sharp=True, norm=True):
        """
        Compare me realization of gray dataset and keras

        :param norm:
        :param sharp:
        :param number_rows: number of rows
        :param number_cols: number of columns
        :return: None
        """

        if sharp and norm:
            train, test = self.dataset['sharped_normal']
        elif sharp:
            train, test = self.dataset['sharped']
        elif norm:
            train, test = self.dataset['normal']
        else:
            train, test = self.dataset['original']

        # My realization:
        print(
            'my realization CNN model for a half of dataset ----------------------------------------------------------')
        my_model = {
            'l1_conv_start': Convolution(num_filters=5, shape=3, image=True),  # 34x34x1 -> 32x32x5
            'l2_pool': MaxPool(size_of_pool=2),  # 32x32x5 -> 16x16x5
            'softmax': Softmax(16 * 16 * 5, 2)}  # 16x16x5 -> 10

        pr = self.my_cnn(train[:500], self.y_train[:500], 5, model=my_model)
        predict_val = pr(test)
        print('accuracy of my CNN model', accuracy_score(predict_val, np.argmax(self.y_test, axis=1)))

        # Keras:
        print(
            'keras realization CNN model with same layers and approximately optimizer method for full dataset --------')
        gray_model = Sequential()
        gray_model.add(Conv2D(5, (3, 3), padding='same', input_shape=(number_rows, number_cols, 1), activation='relu'))
        gray_model.add(MaxPooling2D(pool_size=(2, 2)))
        gray_model.add(Dropout(0.25))

        gray_model.add(Flatten())
        gray_model.add(Dense(2, activation='softmax'))

        self.load_cnn(x_train=train[:, :, :, np.newaxis], x_test=test[:, :, :, np.newaxis],
                      y_train=self.y_train, y_test=self.y_test, model=gray_model,
                      optimizer=SGD(lr=self.learning_rate_const, decay=self.decay_const, momentum=self.momentum_const,
                                    nesterov=True))

    def plotting_stat(self, epochs=15, models=None, sharp=True, norm=True):
        """
        Plotting keras simple and more complex models of different batch size and learning rate

        :param norm:
        :param sharp:
        :param epochs: number of epochs
        :param models: list of CNN keras models with created layers in next format:
        [(model_1, color_plotting, title_legend), ...]

        :return: None
        """

        if sharp and norm:
            train, test = self.dataset['sharped_normal']
        elif sharp:
            train, test = self.dataset['sharped']
        elif norm:
            train, test = self.dataset['normal']
        else:
            train, test = self.dataset['original']

        if models is None:
            models = [(self.model1, 'r', 'Model #1'), (self.model2, 'b', 'Model #2')]

        fig, ax = plt.subplots()

        frame = pd.DataFrame()

        # ---------------------------------------------------------------------------------------------------
        # Plotting keras simple and more complex models of different batch size

        for model, color_of_model, title_of_model in models:
            # get df of all batch sizes and their accuracies
            frame = frame.append(
                self.batch_grid(x_train=train, x_test=test, y_train=self.y_train, y_test=self.y_test, model=model,
                                nb_epoch=epochs, axis=ax, title=title_of_model, color=color_of_model),
                ignore_index=True)

        # sort by batch size descending for bigger value of batch size, get index of the biggest value of accuracy and
        # replace the best value for batch_size_const
        frame = frame.sort_values('batch size', ascending=False)
        batch_size_const = int(frame.loc[frame.accuracy.idxmax(), :]['batch size'])

        # Plotting:
        plt.legend(loc=1)
        plt.title(f'Batch grid for model with {epochs} epoch')
        plt.show()

        # ---------------------------------------------------------------------------------------------------
        # Plotting keras simple and more complex models of different learning_rate

        fig, ax = plt.subplots()

        frame = pd.DataFrame()

        for model, color_of_model, title_of_model in models:
            # get df of all batch sizes and their accuracies
            frame = frame.append(
                self.learning_rate_grid(x_train=train, x_test=test, y_train=self.y_train, y_test=self.y_test,
                                        model=model,
                                        nb_epoch=epochs, axis=ax, title=title_of_model,
                                        batch_s=batch_size_const, color=color_of_model),
                ignore_index=True)

        # sort by learning rate descending for bigger value of learning rate,
        # get index of the biggest value of accuracy and

        # replace the best value for learning_rate_const
        frame = frame.sort_values('learning rate', ascending=False)
        learning_rate_const = frame.loc[frame.accuracy.idxmax(), :]['learning rate']

        # Plotting:
        plt.legend(loc=1)
        plt.title(f'Learning rate grid for model with {epochs} epoch and {batch_size_const} batch size')
        plt.show()

        # ---------------------------------------------------------------------------------------------------
        # Plotting keras simple and more complex models of different learning_rate

        fig, ax = plt.subplots()

        frame = pd.DataFrame()

        for model, color_of_model, title_of_model in models:
            # get df of all batch sizes and their accuracies
            frame = frame.append(
                self.optimizer_grid(x_train=train, x_test=test, y_train=self.y_train, y_test=self.y_test, model=model,
                                    nb_epoch=epochs, axis=ax, title=title_of_model,
                                    batch_s=batch_size_const, color=color_of_model),
                ignore_index=True)

        # sort by learning rate descending for bigger value of learning rate,
        # get index of the biggest value of accuracy and

        # replace the best value for learning_rate_const
        # frame = frame.sort_values('learning rate', ascending=False)
        # learning_rate_const = frame.loc[frame.accuracy.idxmax(), :]['learning rate']

        # Plotting:
        ax.set(xlim=(-0.1, 1.1))
        plt.legend(loc=1)
        plt.title(f'Optimize methods grid for model with {epochs} epoch and {batch_size_const} batch size')
        plt.show()

    def plotting_diff_models(self, row=34, col=34, optimize_method=None, size_range=None, normal=None, sharped=None,
                             model_list=None):
        if size_range is None:
            size_range = [1, 2, 3, 4]
        if normal is None:
            normal = [False, True]
        if sharped is None:
            sharped = [False, True]

        if optimize_method is None:
            optimize_method = 'sgd'

        if model_list is None:
            model_list = [1, 2, 3]

        df_models_acc = pd.DataFrame(columns=['model number', 'accuracy', 'normal', 'sharped', 'size'])

        for size_cof, norm, shr in [(x, y, z) for x in size_range for y in normal for z in sharped]:

            res = self.cnn_diff_models(row=row * size_cof, col=col * size_cof, list_of_models=model_list,
                                       load_flag=False, norm=norm, sharp=shr, optimum_method_name=optimize_method)

            for obj, model_number in zip(res, model_list):
                _, acc = obj
                df_models_acc = df_models_acc.append(
                    {'model number': f'model #{model_number}',
                     'accuracy': acc,
                     'normal': 'normal' if norm else 'not normal',
                     'sharped': 'sharped' if shr else 'not sharped',
                     'size': f'{row * size_cof}x{col * size_cof}'}, ignore_index=True)

        df_pivot = pd.pivot_table(df_models_acc, index=['size', 'model number'], values='accuracy',
                                  columns=['normal', 'sharped'])
        ax = df_pivot.plot(subplots=True, rot=0, figsize=(9, 7), style='.-')
        ax[0].set_title(
            f'Accuracies of models with different parameters: \nsize, normalize, sharped\nOptimum method is '
            f'{optimize_method}')
        plt.show()
        print(df_pivot)

    def metrics_cnn(self, y_pred):

        fpr, tpr, thresholds_rf = roc_curve(np.argmax(self.y_test, axis=1), y_pred[:, 1])
        auc_value = auc(fpr, tpr)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Roc auc (area = {:.3f})'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

        matrix = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1))

        print(pd.DataFrame(matrix,
                           columns=['Predict "No melanoma"', 'Predict "Melanoma"'],
                           index=['Actual "No melanoma"', 'Actual "Melanoma"']))

        tn, fp, fn, tp = matrix.ravel()

        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = 1 - fp / (fp + tn)
        f1_score = 2 * sensitivity * precision / (sensitivity + precision)
        dor = sensitivity * specificity / ((1 - sensitivity) * (1 - specificity))

        print(f'sensitivity: {np.round(sensitivity, 3)}')
        print(f'precision: {np.round(precision, 3)}')
        print(f'specificity: {np.round(specificity, 3)}')
        print(f'f1_score: {np.round(f1_score, 3)}')
        print(f'dor: {np.round(dor, 3)}')


def run(img=None):
    if img is None:
        img = [['ISIC_0001910', 0],
               # ['ISIC_0034287', 1],
               # ['ISIC_0001026', 0],
               # ['ISIC_0033611', 1],
               # ['ISIC_0000091', 0],
               # ['ISIC_0000253', 0],
               # ['ISIC_0012678', 1],
               ['ISIC_0010990', 1]]

    print('Initialize cnn handler:')
    cnn_h = HandlerCNN()

    train, test = cnn_h.dataset['sharped_normal']

    try:
        print('Loading model...')

        f_model = load_model(os.path.join(cnn_h.models_down_dir, 'final_model.h5'))
        scores = f_model.evaluate(test, cnn_h.y_test, verbose=0)
        print("Accuracy on test data: %.2f%%" % (scores[1] * 100))

    except Exception as e:
        print('No final model')
        return -1

    print('------------------------------')
    print('Predicting ...')

    print('Confusion matrix for test data:')
    print(pd.DataFrame(confusion_matrix(np.argmax(cnn_h.y_test, axis=1), np.argmax(f_model.predict(test), axis=1)),
                       columns=['Predict "No melanoma"', 'Predict "Melanoma"'],
                       index=['Actual "No melanoma"', 'Actual "Melanoma"']))

    print('------------------------------\n')
    for name, melanoma in img:
        print(f'Enter img name: {name}.jpg')
        print(f'Label for this img is "{"melanoma" if melanoma else "no melanoma"}"')

        image_i = image.img_to_array(image.load_img(os.path.join(cnn_h.path_to_archive, 'images', f"{name}.jpg"),
                                                    target_size=(cnn_h.rows, cnn_h.cols)))
        val, bins = np.histogram(image_i.flatten(), normed=True)
        val = np.cumsum(np.diff(bins) * val)

        image_i = np.interp(image_i.flatten(), bins[:-1], val * 255).reshape(image_i.shape)
        image_i /= 255

        pr = f_model.predict(np.array([image_i]))
        print(f'Predicted melanoma is {round(float(pr[0][1]) * 100, 2)}%', '\n')


def model_tuning(model_function, param=None):
    if param is None:
        count_list = [32, 32, 64, 64, 150, 150]
        shape_list = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (5, 5)]
        param = zip(count_list, shape_list)

    cnn_h = HandlerCNN()
    cnn_h.schedule_bool = True

    optimum = SGD(lr=cnn_h.learning_rate_const, decay=cnn_h.decay_const,
                  momentum=cnn_h.momentum_const, nesterov=True)
    epoch = 250
    time_df = pd.DataFrame(columns=['variant id', 'time (min)'])

    plt.style.use("ggplot")
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 10), constrained_layout=True)
    loss_max = []
    loss_min = []
    acc_max = []
    acc_min = []
    for i, (count, shape) in enumerate(param):
        model = model_function(count, shape, cnn_h.rows, cnn_h.cols, cnn_h.gray)

        train, test = cnn_h.dataset['sharped_normal']

        print(f'i={i}, count={count}, shape={shape}', end=': ')
        start = t.time()
        m, accuracy = cnn_h.load_cnn(train, test, cnn_h.y_train, cnn_h.y_test, model, optimum, nb_epoch=epoch,
                                     reset_weights_bool=False,
                                     batch_size=None, model_name='', plotting=False, generator=True,
                                     steps_per_epoch=None, logging=0)

        time_df = time_df.append({'variant id': i, 'time (min)': round((t.time() - start) // 60, 2)}, ignore_index=True)
        ax[0, 0].plot(m.epoch, m.history["loss"], label=f"train_loss_{i}", marker='o', linestyle='--', markersize=5)
        ax[0, 1].plot(m.epoch, m.history["val_loss"], label=f"val_loss_{i}, "
                                                            f"last loss {round(m.history['val_loss'][-1], 2)}",
                      marker='o', linestyle='--', markersize=2.9)
        ax[1, 0].plot(m.epoch, m.history["accuracy"], label=f"train_accuracy_{i}", marker='o', linestyle='--',
                      markersize=5)
        ax[1, 1].plot(m.epoch, m.history["val_accuracy"], label=f"val_accuracy_{i}, test acc {accuracy}", marker='o',
                      linestyle='--', markersize=2.9)

        acc_max.append(max(max(m.history["accuracy"]), max(m.history["val_accuracy"])))
        acc_min.append(min(min(m.history["accuracy"]), min(m.history["val_accuracy"])))
        loss_max.append(max(max(m.history["loss"]), max(m.history["val_loss"])))
        loss_min.append(min(min(m.history["loss"]), min(m.history["val_loss"])))

    ax[0, 0].set_ylim(min(loss_min) - 0.01, max(loss_max) + 0.01)
    ax[0, 1].set_ylim(min(loss_min) - 0.01, max(loss_max) + 0.01)
    ax[1, 0].set_ylim(min(acc_min) - 0.01, max(acc_max) + 0.01)
    ax[1, 1].set_ylim(min(acc_min) - 0.01, max(acc_max) + 0.01)

    ax[0, 1].yaxis.set_ticklabels([])
    ax[1, 1].yaxis.set_ticklabels([])
    ax[0, 0].xaxis.set_ticklabels([])
    ax[0, 1].xaxis.set_ticklabels([])

    ax[0, 0].set_ylabel("Loss")
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].set_xlabel("Epoch #")
    ax[1, 1].set_xlabel("Epoch #")
    ax[0, 0].legend(loc='best')
    ax[0, 1].legend(loc='best')
    ax[1, 0].legend(loc='best')
    ax[1, 1].legend(loc='best')
    plt.show()

    return time_df


def model_tuning_compare(compare=True):
    def model1_1(count, shape, rows, cols, gray):
        model = Sequential()
        model.add(Conv2D(count, shape, padding='same', input_shape=(rows, cols, gray), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        return model

    def model1_2(count, shape, rows, cols, gray):
        model = Sequential()

        model.add(Conv2D(count // 2, shape, padding='valid', input_shape=(rows, cols, gray), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(count // 2, shape, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        return model

    def model1_3(count, shape, rows, cols, gray):
        model = Sequential()

        model.add(Conv2D(count // 3, shape, padding='same', input_shape=(rows, cols, gray), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(count // 3, shape, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(count // 3, shape, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        return model

    def model2_1(count, shape, rows, cols, gray):
        model = Sequential()

        model.add(Conv2D(count, shape, padding='same', input_shape=(rows, cols, gray), activation='relu'))
        model.add(Conv2D(count, shape, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        return model

    def model2_2(count, shape, rows, cols, gray):
        model = Sequential()

        model.add(Conv2D(count // 2, shape, padding='same', input_shape=(rows, cols, gray), activation='relu'))
        model.add(Conv2D(count // 2, shape, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(count // 2, shape, padding='same', activation='relu'))
        model.add(Conv2D(count // 2, shape, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        return model

    if compare:
        count_list = [32, 32, 64, 64]
        shape_list = [(3, 3), (5, 5), (3, 3), (5, 5)]
        param = zip(count_list, shape_list)

        model_tuning(model2_1)
        model_tuning(model2_2, param)
    return model1_1, model1_2, model1_3, model2_1, model2_2


if __name__ == '__main__':
    models = model_tuning_compare(compare=False)

    cnn_h = HandlerCNN()
    cnn_h.schedule_bool = False
    cnn_h.early_stop_param = 35
    cnn_h.steps_per_epoch_const = 500
    cnn_h.learning_rate_const = 0.001
    train, test = cnn_h.dataset['sharped_normal']

    optimum = SGD(lr=cnn_h.learning_rate_const, decay=cnn_h.decay_const,
                  momentum=cnn_h.momentum_const, nesterov=True)

    epoch = 200
    cnn_h.load_cnn(train, test, cnn_h.y_train, cnn_h.y_test, cnn_h.model2, optimum, nb_epoch=epoch,
                   reset_weights_bool=False,
                   batch_size=None, model_name='final_model_1', plotting=True, generator=True,
                   steps_per_epoch=None, logging=2)

    epoch = 250
    count, shape = 32, (5, 5)

    model = models[1](count, shape, cnn_h.rows, cnn_h.cols, cnn_h.gray)
    cnn_h.load_cnn(train, test, cnn_h.y_train, cnn_h.y_test, model, optimum, nb_epoch=epoch,
                   reset_weights_bool=False,
                   batch_size=None, model_name='final_model_2', plotting=True, generator=True,
                   steps_per_epoch=None, logging=2)

    model = models[-2](count, shape, cnn_h.rows, cnn_h.cols, cnn_h.gray)
    cnn_h.load_cnn(train, test, cnn_h.y_train, cnn_h.y_test, model, optimum, nb_epoch=epoch,
                   reset_weights_bool=False,
                   batch_size=None, model_name='final_model_3', plotting=True, generator=True,
                   steps_per_epoch=None, logging=2)

    model = models[-1](count, shape, cnn_h.rows, cnn_h.cols, cnn_h.gray)
    cnn_h.load_cnn(train, test, cnn_h.y_train, cnn_h.y_test, model, optimum, nb_epoch=epoch,
                   reset_weights_bool=False,
                   batch_size=None, model_name='final_model_4', plotting=True, generator=True,
                   steps_per_epoch=None, logging=2)
