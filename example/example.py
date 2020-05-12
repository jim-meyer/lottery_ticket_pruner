import collections
import json

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import pandas as pd

import lottery_ticket_pruner

EvaluationResult = collections.namedtuple('EvaluationResult', 'loss accuracy')


class LoggingCheckpoint(keras.callbacks.Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.epoch_data = {}

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        key = '{}_epoch_{}'.format(self.experiment, epoch)
        self.epoch_data[key] = logs


class MNIST(object):
    def __init__(self, experiment):
        self.batch_size = 128
        self.num_classes = 10

        # input image dimensions
        self.img_rows, self.img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

        self.logging_callback = LoggingCheckpoint(experiment)
        self.callbacks = [self.logging_callback]

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def fit(self, model, epochs):
        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test),
                  callbacks=self.callbacks)

    def evaluate(self, model):
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, accuracy

    def get_epoch_logs(self):
        """ Returns a dict of each epoch's accuracy, loss, validation accuracy, validation loss keyed by experiment name+epoch """
        return self.logging_callback.epoch_data


class MNISTNoDropout(MNIST):
    def __init__(self, experiment):
        super().__init__(experiment)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


class MNISTNoDropoutGloballyPruned(MNISTNoDropout):
    def __init__(self, experiment, pruner):
        super().__init__(experiment)
        self.pruner = pruner

    def fit(self, model, epochs):
        callbacks = self.callbacks + [lottery_ticket_pruner.PrunerCallback(self.pruner)]
        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test),
                  callbacks=callbacks)


def compare():
    epochs = 1

    test_results = collections.defaultdict(dict)

    experiment = 'MNIST'
    mnist = MNIST(experiment)
    model = mnist.create_model()
    mnist.fit(model, epochs)
    test_results[experiment]['loss'], test_results[experiment]['accuracy'] = mnist.evaluate(model)
    epoch_logs = mnist.get_epoch_logs()

    experiment = 'MNISTNoDropout'
    mnist_no_dropout = MNISTNoDropout(experiment)
    model = mnist_no_dropout.create_model()
    mnist_no_dropout.fit(model, epochs)
    test_results[experiment]['loss'], test_results[experiment]['accuracy'] = mnist_no_dropout.evaluate(model)
    epoch_logs.update(mnist_no_dropout.get_epoch_logs())

    prune_rate = 0.2
    overall_prune_rate = 0.0
    # Use the weights from the trained model as the basis for determining what weights we'll prune for the new model.
    pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
    for i in range(1):
        prune_rate = pow(prune_rate, 1.0 / (i + 1))
        overall_prune_rate = overall_prune_rate + prune_rate * (1.0 - overall_prune_rate)

        pruner.prune_weights(prune_rate, 'smallest_weights_global')

        experiment = 'MNISTNoDropout_pruned@{:.3f}'.format(overall_prune_rate)
        mnist_pruned = MNISTNoDropoutGloballyPruned(experiment, pruner)
        model = mnist_pruned.create_model()
        mnist_pruned.fit(model, epochs)
        test_results[experiment]['loss'], test_results[experiment]['accuracy'] = mnist_pruned.evaluate(model)
        epoch_logs.update(mnist_pruned.get_epoch_logs())

        # Periodically save the results to allow inspection during these multiple lengthy iterations
        with open('epoch_logs.json', 'w') as f:
            json.dump(epoch_logs, f, indent=4)
        with open('results.json', 'w') as f:
            json.dump(test_results, f, indent=4)

    epoch_logs_df = pd.DataFrame.from_dict(epoch_logs, orient='index')
    epoch_logs_df.to_csv('epoch_logs.csv')
    print(epoch_logs_df)

    results_df = pd.DataFrame.from_dict(test_results, orient='index')
    results_df.to_csv('results.csv')
    print(results_df)


if __name__ == '__main__':
    compare()
