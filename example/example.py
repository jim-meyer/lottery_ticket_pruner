import argparse
import collections
import copy
import json
import os

import keras
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd

import lottery_ticket_pruner


class LoggingCheckpoint(keras.callbacks.Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.epoch_data = {}

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if epoch not in self.epoch_data:
            self.epoch_data[epoch] = collections.OrderedDict()
        self.epoch_data[epoch][self.experiment] = logs


class MNIST(object):
    def __init__(self, experiment, which_set='mnist'):
        self.batch_size = 128
        self.num_classes = 10

        # the data, split between train and test sets
        func_map = {'mnist': self.load_mnist_data, 'cifar10': self.load_cifar10_data, 'cifar10_reduced_10x': self.load_cifar10_reduced_10x_data}
        if which_set in func_map:
            (x_train, y_train), (x_test, y_test) = func_map[which_set]()
        else:
            raise ValueError('`which_set` must be one of {} but it was {}'.format(func_map.keys(), which_set))

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], self.channels, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], self.channels, self.img_rows, self.img_cols)
            self.input_shape = (self.channels, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, self.channels)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, self.channels)
            self.input_shape = (self.img_rows, self.img_cols, self.channels)

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

    def load_cifar10_data(self):
        # input image dimensions
        self.img_rows, self.img_cols = 32, 32
        self.channels = 3

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def load_cifar10_reduced_10x_data(self):
        (x_train, y_train), (x_test, y_test) = self.load_cifar10_data()

        def get_first_n(X, y, classes, n):
            result = []
            # Accumulate the first N samples of each class
            for cls in classes:
                first_n = np.where(y == cls)[0][:n]
                result.extend(first_n)
            # Now put them back into the original order
            result = sorted(result)
            X = X[result]
            y = y[result]
            return X, y

        # Reduce overall size of train, test sets by 10x
        x_train, y_train = get_first_n(x_train, y_train, range(self.num_classes), x_train.shape[0] // (self.num_classes * 10))
        x_test, y_test = get_first_n(x_test, y_test, range(self.num_classes), x_test.shape[0] // (self.num_classes * 10))
        return (x_train, y_train), (x_test, y_test)

    def load_mnist_data(self):
        # input image dimensions
        self.img_rows, self.img_cols = 28, 28
        self.channels = 1
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

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
        """ Returns a dict of each epoch's accuracy, loss, validation accuracy, validation loss.
             result[<epoch>][<experiment>]['loss']
             result[<epoch>][<experiment>]['acc']
             result[<epoch>][<experiment>]['val_loss']
             result[<epoch>][<experiment>]['val_acc']
        """
        return self.logging_callback.epoch_data


class MNISTGloballyPruned(MNIST):
    def __init__(self, experiment, pruner, which_set='mnist'):
        super().__init__(experiment, which_set=which_set)
        self.pruner = pruner

    def fit(self, model, epochs):
        callbacks = self.callbacks + [lottery_ticket_pruner.PrunerCallback(self.pruner)]
        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test),
                  callbacks=callbacks)


def evaluate(which_set, prune_strategy, epochs, output_dir):
    """ Evaluates multiple training approaches:
            A model with randomly initialized weights evaluated with no training having been done
            A model trained from randomly initialized weights
            A model with randomly initialized weights evaluated with no training having been done *but* lottery ticket
                pruning has been done prior to evaluation.
            Several models trained from randomly initialized weights *but* with lottery ticket pruning applied at the
                end of every epoch.
        :returns losses and accuracies for the evaluations. Each are a dict of keyed by experiment name and whose value
            is the loss/accuracy.
    """
    losses = {}
    accuracies = {}

    experiment = 'MNIST'
    mnist = MNIST(experiment, which_set=which_set)
    model = mnist.create_model()
    starting_weights = model.get_weights()
    original_model = copy.deepcopy(model)

    experiment = 'MNIST_no_training'
    losses[experiment], accuracies[experiment] = mnist.evaluate(model)

    experiment = 'MNIST'
    mnist.fit(model, epochs)
    losses[experiment], accuracies[experiment] = mnist.evaluate(model)
    epoch_logs = mnist.get_epoch_logs()

    # Use the weights from the trained model as the basis for determining what weights we'll prune for the new model.
    pruner = lottery_ticket_pruner.LotteryTicketPruner(model, original_model=original_model)

    # Evaluate performance of original model with pruning applied but no training at all
    prune_rate = 0.2
    overall_prune_rate = 0.0
    for i in range(4):
        prune_rate = pow(prune_rate, 1.0 / (i + 1))
        overall_prune_rate = overall_prune_rate + prune_rate * (1.0 - overall_prune_rate)

        pruner.prune_weights(prune_rate, prune_strategy)
        model.set_weights(starting_weights)
        pruner.apply_pruning()

        experiment = 'MNIST_no_training_pruned@{:.3f}'.format(overall_prune_rate)
        losses[experiment], accuracies[experiment] = mnist.evaluate(model)

    pruner.reset_masks()

    # Now train from original weights and prune during training
    prune_rate = 0.2
    overall_prune_rate = 0.0
    for i in range(4):
        prune_rate = pow(prune_rate, 1.0 / (i + 1))
        overall_prune_rate = overall_prune_rate + prune_rate * (1.0 - overall_prune_rate)

        pruner.prune_weights(prune_rate, prune_strategy)
        model.set_weights(starting_weights)
        pruner.apply_pruning()

        experiment = 'MNIST_pruned@{:.3f}'.format(overall_prune_rate)
        mnist_pruned = MNISTGloballyPruned(experiment, pruner, which_set=which_set)
        model = mnist_pruned.create_model()
        mnist_pruned.fit(model, epochs)
        losses[experiment], accuracies[experiment] = mnist_pruned.evaluate(model)

        epoch_logs2 = mnist_pruned.get_epoch_logs()
        for epoch in epoch_logs.keys():
            epoch_logs[epoch].update(epoch_logs2[epoch])

        # Periodically save the results to allow inspection during these multiple lengthy iterations
        with open(os.path.join(output_dir, 'epoch_logs.json'), 'w') as f:
            json.dump(epoch_logs, f, indent=4)

    # Now save csv file so it's easier to compare loss, accuracy across the experiments
    headings = []
    for experiment in epoch_logs[0].keys():
        headings.extend([experiment, '', '', ''])
    sub_headings = ['train_loss', 'train_acc', 'val_loss', 'val_acc'] * len(epoch_logs[0])
    epoch_logs_df = pd.DataFrame([], columns=[headings, sub_headings])
    for epoch in range(len(epoch_logs)):
        row = []
        for experiment in epoch_logs[epoch].keys():
            exp_dict = epoch_logs[i][experiment]
            row.extend([exp_dict['loss'], exp_dict['acc'], exp_dict['val_loss'], exp_dict['val_acc']])
        epoch_logs_df.loc[epoch] = row
    epoch_logs_df.to_csv(os.path.join(output_dir, 'epoch_logs.csv'))

    return losses, accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, required=True, default=1,
        help='How many iterations to do. The final results will be averaged over these iterations.')
    parser.add_argument('--epochs', type=int, required=False, default=20,
        help='How many epochs to train for.')
    parser.add_argument('--which_set', type=str, required=False, default='mnist',
        help='Which data set to use. Must be one of "mnist", "cifar10" or "cifar10_reduced_10x". '
             '"cifar10_reduced_10x" is the cifar10 data set with 10x fewer training, test samples. This is '
             'useful for evaluating these pruning strategies using less data')
    parser.add_argument('--prune_strategy', type=str, required=False, default='smallest_weights_global',
        help='Which pruning strategy to use. Must be one of "random", "smallest_weights", "smallest_weights_global".'
                        'See docs for LotteryTicketPruner.prune_weights() for full details.')
    args = parser.parse_args()

    for i in range(args.iterations):
        output_dir = os.path.join(os.path.dirname(__file__), '{}_{}_{}_{}'.format(args.which_set, args.prune_strategy, args.epochs, 1))
        os.makedirs(output_dir, exist_ok=True)
        losses, accuracies = evaluate(args.which_set, args.prune_strategy, args.epochs, output_dir)

        if i == 0:
            # Only add headings once per every two sub-headings to reduce verbosity
            headings = []
            for key in losses.keys():
                headings.extend([key, ''])
            sub_headings = ['loss', 'acc'] * len(losses)
            results_df = pd.DataFrame([], columns=[headings, sub_headings])
        row = []
        for key in losses.keys():
            row.extend([losses[key], accuracies[key]])
        results_df.loc[i] = row

        results_df.to_csv(os.path.join(output_dir, 'results.csv'))
        print(results_df)

    mean = results_df.mean(axis=0)
    results_df.loc['average'] = row
    results_df.to_csv(os.path.join(output_dir, 'results.csv'))
    print(results_df)
