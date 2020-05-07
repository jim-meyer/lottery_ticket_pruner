import math
import random
import unittest

import keras
import numpy as np
import tensorflow

import lottery_ticket_pruner.lottery_ticket_pruner as lottery_ticket_pruner

TEST_NUM_CLASSES = 3
TEST_DENSE_INPUT_DIMS = (32, )
TEST_DENSE_LAYER_INPUTS = np.prod(TEST_DENSE_INPUT_DIMS)

TEST_DNN_INPUT_DIMS = (64, 64, 3)
TEST_DNN_NUM_CLASSES = 10


class TestLotteryTicketStateManager(unittest.TestCase):
    def _create_test_model(self):
        input = keras.Input(shape=TEST_DENSE_INPUT_DIMS, dtype='float32')
        x = keras.layers.Dense(TEST_NUM_CLASSES)(input)
        model = keras.Model(inputs=input, outputs=x)
        return model

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

    def _summed_model_weights(self, model):
        weights_sum = 0.0
        for layer in model.layers:
            weights = layer.get_weights()
            weights_sum += sum(np.sum(w) for w in weights)
        return weights_sum

    #
    # prune_func_smallest_weights()
    #
    def test_prune_func_smallest_weights(self):
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([1, 2, 3, 4], dtype=float), np.array([1, 1, 1, 1]), prune_percentage=0.25)
        self.assertTrue(np.array_equal([0, 1, 1, 1], actual_mask))

        # Just changed order of weights
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([3, 1, 2, 4], dtype=float), np.array([1, 1, 1, 1]), prune_percentage=0.5)
        self.assertTrue(np.array_equal([1, 0, 0, 1], actual_mask))

        # Odd number of weights
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([5, 3, 1, 2, 4], dtype=float), np.array([1, 1, 1, 1, 1]), prune_percentage=0.5)
        self.assertTrue(np.array_equal([1, 1, 0, 0, 1], actual_mask))

        # Current mask masks out one of the lowest weights
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([1, 2, 3, 4, 5], dtype=float), np.array([0, 1, 1, 1, 1]), prune_percentage=0.25)
        self.assertTrue(np.array_equal([0, 0, 1, 1, 1], actual_mask))

        # Current mask masks out one of the lowest weights
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([1, 2, 3, 4], dtype=float), np.array([0, 1, 1, 0]), prune_percentage=0.25)
        self.assertTrue(np.array_equal([0, 0, 1, 0], actual_mask))

        # Some negative and some positive weights should be masked
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([-1, 2, -3, 4], dtype=float), np.array([1, 1, 1, 1]), prune_percentage=0.5)
        self.assertTrue(np.array_equal([0, 0, 1, 1], actual_mask))

        # Many identical values but only some of them should get masked
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([1, 1, 1, 1, 2, 2], dtype=float), np.array([1, 1, 1, 1, 1, 1]), prune_percentage=0.5)
        self.assertEqual(3, np.sum(actual_mask))

        # Many identical absolute values but only some of them should get masked
        actual_mask = lottery_ticket_pruner.prune_func_smallest_weights(np.array([]), np.array([1, -1, -1, 1, 2, -2], dtype=float), np.array([1, 1, 1, 1, 1, 1]), prune_percentage=0.5)
        self.assertEqual(3, np.sum(actual_mask))

    #
    # LotteryTicketPrunerRandom
    #
    def test_LotteryTicketPrunerRandom(self):
        model = self._create_test_model()
        mgr = lottery_ticket_pruner.LotteryTicketPruner(model)

        # First layer is the input layer; ignore it
        # Second layer is Dense layer with 2 weights. First is fully connected weights. Second is output weights.
        interesting_key = tuple([model.layers[1], tuple([0])])
        num_unmasked = np.sum(mgr.prune_masks_map[interesting_key][0])
        self.assertEqual(num_unmasked, TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES)

        # No pruning percent specified so no weight should change
        initial_model_weights_sum = self._summed_model_weights(model)
        mgr.apply_pruning()
        new_model_weights_sum = self._summed_model_weights(model)
        self.assertEqual(initial_model_weights_sum, new_model_weights_sum)

        mgr.prune_weights(0.5, local_prune_func=lottery_ticket_pruner.prune_func_random)
        num_masked = np.sum(mgr.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.5))

        mgr.prune_weights(0.2, local_prune_func=lottery_ticket_pruner.prune_func_random)
        num_masked = np.sum(mgr.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.6))

        mgr.restore_initial_weights()
        new_model_weights_sum = self._summed_model_weights(model)
        self.assertEqual(initial_model_weights_sum, new_model_weights_sum)

        mgr.apply_pruning()
        num_masked = np.sum(mgr.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.6))

    #
    # LotteryTicketPrunerSmallestWeights
    #
    def test_LotteryTicketPrunerSmallestWeights(self):
        model = self._create_test_model()
        # First layer is the input layer; ignore it
        # Second layer is Dense layer with 2 weights. First is fully connected weights. Second is output weights.
        interesting_layer = model.layers[1]
        interesting_layer_shape = interesting_layer.weights[0].shape
        interesting_layer_weight_count = int(np.prod(interesting_layer_shape))
        interesting_key = tuple([model.layers[1], tuple([0])])

        dl_test_weights = np.random.choice(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES, size=TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES, replace=False)
        # Get rid of zero weights since we count those below during verification
        dl_test_weights += 1
        dl_test_weights = dl_test_weights.reshape(interesting_layer_shape)
        interesting_layer.set_weights([dl_test_weights, interesting_layer.get_weights()[1]])
        mgr = lottery_ticket_pruner.LotteryTicketPruner(model)

        mgr.prune_weights(0.5, local_prune_func=lottery_ticket_pruner.prune_func_smallest_weights)
        actual_weights = interesting_layer.get_weights()
        actual_weights[0][actual_weights[0] == 0.0] = math.inf
        min_weight = np.min(actual_weights[0])
        self.assertGreaterEqual(min_weight, int(interesting_layer_weight_count * 0.5))

        num_masked = np.sum(mgr.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.5))

        mgr.prune_weights(0.2, local_prune_func=lottery_ticket_pruner.prune_func_smallest_weights)
        actual_weights = interesting_layer.get_weights()
        actual_weights[0][actual_weights[0] == 0.0] = math.inf
        min_weight = np.min(actual_weights[0])
        self.assertGreaterEqual(min_weight, int(interesting_layer_weight_count * 0.6))

        num_masked = np.sum(mgr.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.6))

    def test_LotteryTicketPrunerSmallestWeights_2(self):
        model = self._create_test_model()
        # First layer is the input layer; ignore it
        # Second layer is Dense layer with 2 weights. First is fully connected weights. Second is output weights.
        interesting_layer = model.layers[1]
        interesting_layer_shape = interesting_layer.weights[0].shape

        dl_test_weights = np.random.choice(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES, size=TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES, replace=False)
        # Make some weights negative
        dl_test_weights -= TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES // 2
        dl_test_weights = dl_test_weights.reshape(interesting_layer_shape)
        interesting_layer.set_weights([dl_test_weights, interesting_layer.get_weights()[1]])
        mgr = lottery_ticket_pruner.LotteryTicketPruner(model)

        prune_rate = 0.5
        mgr.prune_weights(prune_rate, local_prune_func=lottery_ticket_pruner.prune_func_smallest_weights)
        actual_weights = interesting_layer.get_weights()
        min_expected_pos = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate // 2 - 1
        max_expected_neg = -TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate // 2 + 1
        unpruned_pos = np.sum(actual_weights[0] >= min_expected_pos)
        unpruned_neg = np.sum(actual_weights[0] <= max_expected_neg)
        unpruned = unpruned_pos + unpruned_neg
        self.assertIn(unpruned, [int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate), int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate) - 1])
        expected_to_be_pruned = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES - unpruned - 1
        self.assertEqual(int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate), expected_to_be_pruned)

        # Prune again
        prune_rate2 = 0.1
        expected_to_be_pruned2 = int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate2 * (1.0 - prune_rate))
        mgr.prune_weights(prune_rate2, local_prune_func=lottery_ticket_pruner.prune_func_smallest_weights)
        actual_weights = interesting_layer.get_weights()
        min_expected_pos = expected_to_be_pruned2 // 2 - 1
        max_expected_neg = -expected_to_be_pruned2 // 2 + 1
        unpruned_pos = np.sum(actual_weights[0] >= min_expected_pos)
        unpruned_neg = np.sum(actual_weights[0] <= max_expected_neg)
        unpruned = unpruned_pos + unpruned_neg
        expected_unpruned = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES - expected_to_be_pruned - expected_to_be_pruned2
        self.assertLessEqual(abs(expected_unpruned - unpruned), 1)

    def test_LotteryTicketPrunerSmallestWeights_SimilarWeights(self):
        """ Tests case where many or all weights are same value. Hence we might be tempted to mask on all of the
        smallest weights rather than honoring only up to the prune rate
        """
        model = self._create_test_model()
        # First layer is the input layer; ignore it
        # Second layer is Dense layer with 2 weights. First is fully connected weights. Second is output weights.
        interesting_layer = model.layers[1]
        interesting_layer_shape = interesting_layer.weights[0].shape

        # Make all weights the same
        dl_test_weights = np.ones([TEST_DENSE_LAYER_INPUTS, TEST_NUM_CLASSES], dtype=int)
        # Make some weights negative
        dl_test_weights = dl_test_weights.reshape(interesting_layer_shape)
        interesting_layer.set_weights([dl_test_weights, interesting_layer.get_weights()[1]])
        mgr = lottery_ticket_pruner.LotteryTicketPruner(model)

        prune_rate = 0.5
        mgr.prune_weights(prune_rate, local_prune_func=lottery_ticket_pruner.prune_func_smallest_weights)
        actual_weights = interesting_layer.get_weights()
        expected = int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate)
        actual = np.sum(actual_weights[0])
        self.assertEqual(expected, actual)

    def test_LotteryTicketPrunerSmallestWeights_GlobalPruning(self):
        """ Tests case where many or all weights are same value. Hence we might be tempted to mask on all of the
        smallest weights rather than honoring only up to the prune rate
        """
        random.seed(1234)
        np.random.seed(2345)
        tensorflow.set_random_seed(3456)

        model = self._create_test_dnn_model()
        interesting_layers = [model.layers[1], model.layers[4], model.layers[8]]
        interesting_weight_index = 0

        # Make sure no weights are zero so our checks below for zeroes only existing in masked weights is reliable
        weight_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            weights[interesting_weight_index][weights[interesting_weight_index] == 0.0] = 0.1234
            layer.set_weights(weights)
            num_weights = np.prod(weights[interesting_weight_index].shape)
            weight_counts.append(num_weights)

        mgr = lottery_ticket_pruner.LotteryTicketPruner(model)

        prune_rate = 0.5
        mgr.prune_weights(prune_rate, global_prune_func=lottery_ticket_pruner.prune_func_smallest_weights_global)

        pruned_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            pruned_counts.append(np.sum(weights[interesting_weight_index] == 0.0))

        total_weights = np.sum(weight_counts)
        num_pruned = np.sum(pruned_counts)
        self.assertAlmostEqual(prune_rate, num_pruned / total_weights, places=1)
        # Given the seeding we did at the beginning of this test these results should be reproducible. They were
        # obtained by manual inspection.
        self.assertEqual(67, pruned_counts[0])
        self.assertEqual(3, pruned_counts[1])
        self.assertEqual(5, pruned_counts[2])

        # Now prune once more to make sure cumulative pruning works as expected
        total_prune_rate = prune_rate
        prune_rate = 0.2
        total_prune_rate = total_prune_rate + (1.0 - total_prune_rate) * prune_rate
        mgr.prune_weights(prune_rate, global_prune_func=lottery_ticket_pruner.prune_func_smallest_weights_global)

        pruned_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            pruned_counts.append(np.sum(weights[interesting_weight_index] == 0.0))

        total_weights = np.sum(weight_counts)
        num_pruned = np.sum(pruned_counts)
        self.assertEqual(num_pruned / total_weights, total_prune_rate)
        # Given the seeding we did at the beginning of this test these results should be reproducible. They were
        # obtained by manual inspection.
        self.assertEqual(80, pruned_counts[0])
        self.assertEqual(4, pruned_counts[1])
        self.assertEqual(6, pruned_counts[2])


if __name__ == '__main__':
    unittest.main()
