# This test does specific comparisons that depend on the random seeds being set the same from run to run
import random
random.seed(1234)

import numpy as np  # noqa
np.random.seed(2345)

# Dancing needed to work with TF 1.x and 2.x
import tensorflow   # noqa
if hasattr(tensorflow, 'set_random_seed'):
    tensorflow.set_random_seed(3456)
else:
    tensorflow.random.set_seed(3456)

import unittest     # noqa

import numpy as np  # noqa
import tensorflow.keras as keras    # noqa

import lottery_ticket_pruner    # noqa

TEST_DNN_INPUT_DIMS = (64, 64, 3)
TEST_DNN_NUM_CLASSES = 10


class TestLotteryTicketPrunerRandseed(unittest.TestCase):
    def _create_test_dnn_model(self):
        input = keras.Input(shape=TEST_DNN_INPUT_DIMS, dtype='float32')
        x = keras.layers.Conv2D(4,
                                kernel_size=3,
                                strides=(2, 2),
                                padding='valid',
                                use_bias=True,
                                name='Conv1')(input)
        x = keras.layers.BatchNormalization(axis=1,
                                            epsilon=1e-3,
                                            momentum=0.999,
                                            name='bn_Conv1')(x)
        x = keras.layers.ReLU(6., name='Conv1_relu')(x)

        x = keras.layers.Conv2D(3,
                                kernel_size=1,
                                padding='same',
                                use_bias=False,
                                activation=None,
                                name='Conv2')(x)
        x = keras.layers.BatchNormalization(axis=1,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name='bn_Conv2')(x)
        x = keras.layers.ReLU(6., name='Conv2_relu')(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(TEST_DNN_NUM_CLASSES, activation='softmax',
                               use_bias=True, name='Logits')(x)
        model = keras.Model(inputs=input, outputs=x)
        return model

    #
    # calc_prune_mask()
    #   'smallest_weights_global'
    #
    def test_smallest_weights_global(self):
        """ Tests case where many or all weights are same value. Hence we might be tempted to mask on all of the
        smallest weights rather than honoring only up to the prune rate
        """
        model = self._create_test_dnn_model()
        interesting_layers = [model.layers[1], model.layers[4], model.layers[8]]
        interesting_weights_index = 0

        # Make sure no weights are zero so our checks below for zeroes only existing in masked weights are reliable
        weight_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            weights[interesting_weights_index][weights[interesting_weights_index] == 0.0] = 0.1234
            layer.set_weights(weights)
            num_weights = np.prod(weights[interesting_weights_index].shape)
            weight_counts.append(num_weights)

        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        num_pruned1 = 0
        for layer in interesting_layers:
            weights = layer.get_weights()
            num_pruned1 += np.sum(weights[interesting_weights_index] == 0.0)

        prune_rate = 0.5
        pruner.calc_prune_mask(model, prune_rate, 'smallest_weights_global')

        # calc_prune_mask() shouldn't do the actual pruning so verify that weights didn't change
        num_pruned2 = 0
        for layer in interesting_layers:
            weights = layer.get_weights()
            num_pruned2 += np.sum(weights[interesting_weights_index] == 0.0)
        self.assertEqual(num_pruned1, num_pruned2)

        pruner.apply_pruning(model)
        pruned_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            pruned_counts.append(np.sum(weights[interesting_weights_index] == 0.0))

        total_weights = np.sum(weight_counts)
        num_pruned = np.sum(pruned_counts)
        self.assertAlmostEqual(prune_rate, num_pruned / total_weights, places=1)
        # Given the seeding we did at the beginning of this test these results should be reproducible. They were
        # obtained by manual inspection.
        # Ranges are used here since TF 1.x on python 3.6, 3.7 gives slightly different results from TF 2.x on
        # python 3.8. These assertions accomodate both.
        self.assertTrue(62 <= pruned_counts[0] <= 67, msg=f'pruned_counts={pruned_counts}')
        self.assertTrue(2 <= pruned_counts[1] <= 5, msg=f'pruned_counts={pruned_counts}')
        self.assertTrue(5 <= pruned_counts[2] <= 9, msg=f'pruned_counts={pruned_counts}')
        self.assertEqual(75, sum(pruned_counts))

        # Now prune once more to make sure cumulative pruning works as expected
        total_prune_rate = prune_rate
        prune_rate = 0.2
        total_prune_rate = total_prune_rate + (1.0 - total_prune_rate) * prune_rate
        pruner.calc_prune_mask(model, prune_rate, 'smallest_weights_global')
        pruner.apply_pruning(model)

        pruned_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            pruned_counts.append(np.sum(weights[interesting_weights_index] == 0.0))

        total_weights = np.sum(weight_counts)
        num_pruned = np.sum(pruned_counts)
        self.assertEqual(num_pruned / total_weights, total_prune_rate)
        # Given the seeding we did at the beginning of this test these results should be reproducible. They were
        # obtained by manual inspection.
        # Ranges are used here since TF 1.x on python 3.6, 3.7 gives slightly different results from TF 2.x on
        # python 3.8. These assertions accomodate both.
        self.assertTrue(74 <= pruned_counts[0] <= 78, msg=f'pruned_counts={pruned_counts}')
        self.assertTrue(2 <= pruned_counts[1] <= 5, msg=f'pruned_counts={pruned_counts}')
        self.assertTrue(9 <= pruned_counts[2] <= 12, msg=f'pruned_counts={pruned_counts}')
        self.assertEqual(90, sum(pruned_counts))


if __name__ == '__main__':
    unittest.main()
