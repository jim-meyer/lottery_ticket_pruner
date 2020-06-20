import unittest

import tensorflow.keras as keras
import numpy as np

import lottery_ticket_pruner
from examples.example import MNIST

TEST_PRUNE_RATE = 0.5


class _VerificationCallback(keras.callbacks.Callback):
    """ Does verifications for these tests """
    def __init__(self, testcase):
        super().__init__()
        self.testcase = testcase

    def on_train_end(self, logs=None):
        """ This callback will be called after the PrunerCallback so we can verify that pruning occurred here """
        super().on_train_end(logs)
        self.testcase._assert_weights_have_been_pruned()

    def on_epoch_begin(self, epoch, logs=None):
        """ This callback will be called after the PrunerCallback so we can verify that pruning occurred here """
        super().on_epoch_begin(epoch, logs)
        self.testcase._assert_weights_have_been_pruned()


class MNISTTest(MNIST):
    def __init__(self, ):
        super().__init__('')
        self.pruner = None
        self.test_verification_callback = None

    def init(self, pruner, test_verification_callback):
        self.pruner = pruner
        self.test_verification_callback = test_verification_callback

    def fit(self, model, epochs):
        callbacks = self.callbacks + [lottery_ticket_pruner.PrunerCallback(self.pruner)]
        # Make sure the test verification callback gets called last by adding it at end of list
        callbacks = callbacks + [self.test_verification_callback]
        # Only use the first 100, 10 samples for training, validation to speed things up for unit test purposes
        model.fit(self.dataset.x_train[:100], self.dataset.y_train[:100],
                  batch_size=128,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(self.dataset.x_test[:10], self.dataset.y_test[:10]),
                  callbacks=callbacks)


class TestKerasPrunerCallback(unittest.TestCase):
    def setUp(self):
        self.mnist_test = MNISTTest()
        self.model = self.mnist_test.create_model()
        self.pruner = lottery_ticket_pruner.LotteryTicketPruner(self.model)

    def _assert_weights_have_been_pruned(self):
        for tpl, layer, index, original_weights, pretrained_weights, current_weights, current_mask in\
                self.pruner.iterate_prunables(self.model):
            # Verify weights
            pruned_count = np.sum(current_weights == 0.0)
            self.assertEqual(np.prod(current_weights.shape) * TEST_PRUNE_RATE, pruned_count)

            # Verify mask
            pruned_count = np.sum(current_mask == False)    # noqa
            self.assertEqual(np.prod(current_mask.shape) * TEST_PRUNE_RATE, pruned_count)

    def test_callback(self):
        epochs = 2
        # Can't do this in constructor of MNISTTest() since we have a chicken and egg problem
        self.mnist_test.init(self.pruner, _VerificationCallback(self))
        self.pruner.calc_prune_mask(self.model, TEST_PRUNE_RATE, 'smallest_weights')

        self.mnist_test.fit(self.model, epochs)
        self._assert_weights_have_been_pruned()


if __name__ == '__main__':
    unittest.main()
