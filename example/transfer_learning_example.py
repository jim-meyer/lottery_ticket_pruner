"""
An example of how to evaluate a classically trained CNN and a CNN trained using Lottery Ticket pruning.
This same can be readily adapted for other models including DNNs.
"""
import argparse
import collections
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
    """ A keras checkpoint that is used during training to capture the test and validation losses and accuracies.
    This allows comparison between normal (unpruned) training and lottery ticket pruning on an epoch by epoch basis.
    """
    def __init__(self, experiment):
        super().__init__()
        self.reset(experiment)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if epoch not in self.epoch_data:
            self.epoch_data[epoch] = collections.OrderedDict()
        self.epoch_data[epoch][self.experiment] = logs

    def reset(self, experiment):
        self.experiment = experiment
        self.epoch_data = {}


class Dataset(object):
    def __init__(self, which_set='mnist'):
        # The MNIST and CIFAR10 datasets each have 10 classes
        self.num_classes = 10
        if which_set is not None:
            # the data, split between train and test sets
            func_map = {'mnist': self.load_mnist_data, 'cifar10': self.load_cifar10_data,
                        'cifar10_reduced_10x': self.load_cifar10_reduced_10x_data}
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
        else:
            self.input_shape = None
            self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def load_cifar10_data(self):
        """ Initialize the instance for training with the full CIFAR dataset.
        :returns The CIFAR dataset divided up into training, testing samples.
            (X train, y train, X test, y test)
        """
        # input image dimensions
        self.img_rows, self.img_cols = 32, 32
        self.channels = 3

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def load_cifar10_reduced_10x_data(self):
        """ Initialize the instance for training with 1/10th of the samples in the CIFAR dataset.
        :returns 1/10th of the CIFAR dataset divided up into training, testing samples.
            (X train, y train, X test, y test)
        """
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
        x_train, y_train = get_first_n(x_train, y_train, range(self.num_classes),
                                       x_train.shape[0] // (self.num_classes * 10))
        x_test, y_test = get_first_n(x_test, y_test, range(self.num_classes),
                                     x_test.shape[0] // (self.num_classes * 10))
        return (x_train, y_train), (x_test, y_test)

    def load_mnist_data(self):
        """ Initialize the instance for training with the MNIST dataset.
        :returns The MNIST dataset divided up into training, testing samples.
            (X train, y train, X test, y test)
        """
        # input image dimensions
        self.img_rows, self.img_cols = 28, 28
        self.channels = 1
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def split_dataset(self):
        """ Splits this dataset into two datasets. The first dataset (self) has the first N/2 classes and the
        second dataset (return value) has the remaining N/2 classes.
        """
        new_dataset = Dataset(which_set=None)
        y_train_categorical = np.argmax(self.y_train, axis=1)
        y_test_categorical = np.argmax(self.y_test, axis=1)
        new_num_classes = self.num_classes // 2

        new_dataset.num_classes = self.num_classes - new_num_classes
        self.num_classes = new_num_classes
        new_dataset.input_shape = self.input_shape

        # Split the training data
        self_train = y_train_categorical < new_num_classes
        new_train = y_train_categorical >= new_num_classes
        new_dataset.x_train = self.x_train[new_train]
        new_dataset.y_train = self.y_train[new_train, new_num_classes:]
        self.x_train = self.x_train[self_train]
        self.y_train = self.y_train[self_train, :new_num_classes]

        # Split the testing data
        self_test = y_test_categorical < new_num_classes
        new_test = y_test_categorical >= new_num_classes
        new_dataset.x_test = self.x_test[new_test]
        new_dataset.y_test = self.y_test[new_test, new_num_classes:]
        self.x_test = self.x_test[self_test]
        self.y_test = self.y_test[self_test, :new_num_classes]

        return new_dataset


class MNIST(object):
    """ A class that can be used to create, train and evaluate a model on the MNIST or CIFAR10 datasets.
    """
    def __init__(self, experiment, which_set='mnist'):
        self.batch_size = 128
        self.dataset = Dataset(which_set=which_set)

        self.experiment = experiment
        self.logging_callback = LoggingCheckpoint(experiment)
        self.callbacks = [self.logging_callback]

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.dataset.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.dataset.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def fit(self, model, epochs, dataset=None):
        if dataset is None:
            dataset = self.dataset
        self.logging_callback.reset(self.experiment)
        model.fit(dataset.x_train, dataset.y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(dataset.x_test, dataset.y_test),
                  callbacks=self.callbacks)

    def evaluate(self, model, dataset=None):
        if dataset is None:
            dataset = self.dataset
        loss, accuracy = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        return loss, accuracy

    def get_epoch_logs(self):
        """ Returns a dict of each epoch's accuracy, loss, validation accuracy, validation loss.
             result[<epoch>][<experiment>]['loss']
             result[<epoch>][<experiment>]['acc']
             result[<epoch>][<experiment>]['val_loss']
             result[<epoch>][<experiment>]['val_acc']
        """
        return self.logging_callback.epoch_data


class MNISTPruned(MNIST):
    """ A class that can be used to create, train and evaluate a model on the MNIST or CIFAR10 datasets.
    When training the model this class will apply lottery ticket pruning after every epoch.
    """
    def __init__(self, experiment, pruner, use_dwr=False, which_set='mnist'):
        """
        :param experiment: A string that describes the experiment being evaluated.
        :param pruner: A `LotteryTicketPruner` instance that is used to prune the weights after every epoch of training.
        :param use_dwr: If True then the callback will apply Dynamic Weight Rescaling (DWR) to the unpruned weights in
            the model after every epoch.
            See section 5.2, "Dynamic Weight Rescaling" of https://arxiv.org/pdf/1905.01067.pdf.
            A quote from that paper describes it best:
                "For each training iteration and for each layer, we multiply the underlying weights by the ratio of the
                total number of weights in the layer over the number of ones in the corresponding mask."
        :param which_set: One of 'mnist', 'cifar10', 'cifar10_reduced_10x'.
            See `evaluate()` for more info on what these mean.
        """
        super().__init__(experiment, which_set=which_set)
        self.pruner = pruner
        self.use_dwr = use_dwr

    def fit(self, model, epochs, dataset=None):
        if dataset is None:
            dataset = self.dataset
        callbacks = self.callbacks + [lottery_ticket_pruner.PrunerCallback(self.pruner, use_dwr=self.use_dwr)]
        model.fit(dataset.x_train, dataset.y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(dataset.x_test, dataset.y_test),
                  callbacks=callbacks)


def evaluate(which_set, prune_strategy, use_dwr, epochs, output_dir):
    """ Evaluates multiple training approaches:
            A model with randomly initialized weights evaluated with no training having been done
            A model trained from randomly initialized weights
            A model with randomly initialized weights evaluated with no training having been done *but* lottery ticket
                pruning has been done prior to evaluation.
            Several models trained from randomly initialized weights *but* with lottery ticket pruning applied at the
                end of every epoch.
        :param which_set: One of 'mnist', 'cifar10', 'cifar10_reduced_10x'.
            'mnist' is the standard MNIST data set (70k total images of digits 0-9).
            'cifar10' is the standard CIFAR10 data set (60k total images in 10 classes, 6k images/class)
            'cifar10_reduced_10x' is just like 'cifar10' but with the total training, test sets reduced by 10x.
                (6k total images in 10 classes, 600 images/class).
                This is useful for seeing the effects of lottery ticket pruning on a smaller dataset.
        :param prune_strategy: One of the strategies supported by `LotteryTicketPruner.calc_prune_mask()`
            A string indicating how the pruning should be done.
                'random': Pruning is done randomly across all prunable layers.
                'smallest_weights': The smallest weights at each prunable layer are pruned. Each prunable layer has the
                    specified percentage pruned from the layer's weights.
                'smallest_weights_global': The smallest weights across all prunable layers are pruned. Some layers may
                    have substantially more or less weights than `prune_percentage` pruned from them. But overall,
                    across all prunable layers, `prune_percentage` weights will be pruned.
                'large_final': Keeps the weights that have the largest magnitude from the previously trained model.
                    This is 'large_final' as defined in https://arxiv.org/pdf/1905.01067.pdf
                'large_final': Keeps the weights that have the largest magnitude from the previously trained model.
                    This is 'large_final' as defined in https://arxiv.org/pdf/1905.01067.pdf
                'large_final_same_sign': TODO - "same sign" logic needs to be applied once to the model prior to
                    training, not during pruning.
        :param boolean use_dwr: Whether or not to apply Dynamic Weight Rescaling (DWR) to the unpruned weights in the
            model.
            See section 5.2, "Dynamic Weight Rescaling" of https://arxiv.org/pdf/1905.01067.pdf.
            A quote from that paper describes it best:
                "For each training iteration and for each layer, we multiply the underlying weights by the ratio of the
                total number of weights in the layer over the number of ones in the corresponding mask."
        :param epochs: The number of epochs to train the models for.
        :param output_dir: The directory to put output files.
        :returns losses and accuracies for the evaluations. Each are a dict of keyed by experiment name and whose value
            is the loss/accuracy.
    """
    losses = {}
    accuracies = {}

    experiment = 'MNIST_xfer_learn'
    mnist = MNIST(experiment, which_set=which_set)
    # Split the dataset into two, the dataset that we'll use to classically train a model, and the dataset we'll
    # use to apply train a new model using transfer learning and lottery ticket pruning.
    tl_dataset = mnist.dataset.split_dataset()
    model = mnist.create_model()

    experiment = 'MNIST_xfer_learn_no_training'
    losses[experiment], accuracies[experiment] = mnist.evaluate(model)

    # Classically train a model on data from half of the classes
    experiment = 'MNIST_xfer_learn_train_1st_half_data'
    mnist.fit(model, epochs)
    losses[experiment], accuracies[experiment] = mnist.evaluate(model)
    # For this experiment we consider the starting weights to be the initial weights of the trained model on the
    # first N/2 class' samples. This is the source model that we will use to do transfer learning to train a model on
    # the remaining data that has "new" class labels.
    starting_weights = model.get_weights()

    pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

    # Now we classically train a model on the other half of the data from the previously unknown classes
    experiment = 'MNIST_xfer_learn_train_2nd_half_data'
    mnist.fit(model, epochs, dataset=tl_dataset)
    trained_weights = model.get_weights()
    losses[experiment], accuracies[experiment] = mnist.evaluate(model, dataset=tl_dataset)
    epoch_logs = mnist.get_epoch_logs()
    pruner.set_pretrained_weights(model)

    # Evaluate performance of model with original weights and pruning applied
    num_prune_rounds = 4
    prune_rate = 0.2
    overall_prune_rate = 0.0
    for i in range(num_prune_rounds):
        prune_rate = pow(prune_rate, 1.0 / (i + 1))
        overall_prune_rate = overall_prune_rate + prune_rate * (1.0 - overall_prune_rate)

        # Make sure each iteration of pruning uses that same trained weights to determine pruning mask
        model.set_weights(trained_weights)
        pruner.calc_prune_mask(model, prune_rate, prune_strategy)
        # Now revert model to original random starting weights and apply pruning
        model.set_weights(starting_weights)
        pruner.apply_pruning(model)

        experiment = 'MNIST_xfer_learn_no_training_pruned@{:.3f}'.format(overall_prune_rate)
        losses[experiment], accuracies[experiment] = mnist.evaluate(model)

    pruner.reset_masks()

    # Calculate pruning mask below using trained weights
    model.set_weights(trained_weights)

    # Now train from original weights and prune during training
    prune_rate = 0.2
    overall_prune_rate = 0.0
    for i in range(num_prune_rounds):
        prune_rate = pow(prune_rate, 1.0 / (i + 1))
        overall_prune_rate = overall_prune_rate + prune_rate * (1.0 - overall_prune_rate)

        # Calculate the pruning mask using the trained model and it's final trained weights
        pruner.calc_prune_mask(model, prune_rate, prune_strategy)

        # Now create a new model that has the original random starting weights and train it
        experiment = 'MNIST_xfer_learn_pruned@{:.3f}'.format(overall_prune_rate)
        mnist_pruned = MNISTPruned(experiment, pruner, use_dwr=use_dwr, which_set=which_set)
        # Need to split the dataset here so `mnist` and `mnist_pruned` models have same shape
        _ = mnist_pruned.dataset.split_dataset()
        prune_trained_model = mnist_pruned.create_model()
        prune_trained_model.set_weights(starting_weights)
        mnist_pruned.fit(prune_trained_model, epochs, dataset=tl_dataset)
        losses[experiment], accuracies[experiment] = mnist_pruned.evaluate(prune_trained_model, dataset=tl_dataset)

        epoch_logs2 = mnist_pruned.get_epoch_logs()
        assert len(epoch_logs) == epochs
        for epoch in range(epochs):
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
    assert len(epoch_logs) == epochs
    for epoch in range(epochs):
        row = []
        for experiment in epoch_logs[epoch].keys():
            exp_dict = epoch_logs[epoch][experiment]
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
                        'See docs for LotteryTicketPruner.calc_prune_mask() for full details.')
    parser.add_argument('--dwr', action='store_true',
        help='Dynamice Weight Rescaling. Specify this to have unpruned weights rescaled after pruning is done.')
    args = parser.parse_args()

    base_output_dir = os.path.dirname(__file__)

    for i in range(args.iterations):
        output_dir = os.path.join(base_output_dir, '{}_xfer_learn_{}_{}_{}'.format(args.which_set, args.prune_strategy,
                                                                                   args.epochs, i))
        os.makedirs(output_dir, exist_ok=True)
        losses, accuracies = evaluate(args.which_set, args.prune_strategy, args.dwr, args.epochs, output_dir)

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

        results_df.to_csv(os.path.join(base_output_dir,
                                       '{}_{}{}_{}_results.csv'.format(args.which_set, args.prune_strategy,
                                                                       '_dwr' if args.dwr else '', args.epochs)))
        print(results_df)

    mean = results_df.mean(axis=0)
    results_df.loc['average'] = mean
    results_df.to_csv(os.path.join(base_output_dir, '{}_{}{}_{}_results.csv'.format(args.which_set, args.prune_strategy,
                                                                                    '_dwr' if args.dwr else '',
                                                                                    args.epochs)))
    print(results_df)
