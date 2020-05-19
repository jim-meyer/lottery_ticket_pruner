""" Copyright (C) 2020 Jim Meyer <jimm@racemed.com> """
import logging
import math
import random
import sys
import unittest

import keras
import numpy as np
import tensorflow

import lottery_ticket_pruner
from lottery_ticket_pruner.lottery_ticket_pruner import _prune_func_smallest_weights, _prune_func_smallest_weights_global

TEST_NUM_CLASSES = 3
TEST_DENSE_INPUT_DIMS = (32, )
TEST_DENSE_LAYER_INPUTS = np.prod(TEST_DENSE_INPUT_DIMS)
TEST_DENSE_WEIGHT_COUNT = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES

TEST_DNN_INPUT_DIMS = (64, 64, 3)
TEST_DNN_NUM_CLASSES = 10


def enable_debug_logging():
    logger = logging.getLogger('lottery_ticket_pruner')
    logger.setLevel('DEBUG')
    logger.addHandler(logging.StreamHandler(sys.stdout))

# enable_debug_logging()


class TestLotteryTicketStateManager(unittest.TestCase):
    def _create_test_model(self):
        input = keras.Input(shape=TEST_DENSE_INPUT_DIMS, dtype='float32')
        x = keras.layers.Dense(TEST_NUM_CLASSES)(input)
        model = keras.Model(inputs=input, outputs=x)
        return model

    def _create_test_model_diff_shape(self, diff_input_shape=False, diff_output_shape=False):
        input_dims = (64, ) if diff_input_shape else TEST_DENSE_INPUT_DIMS
        output_dims = (TEST_NUM_CLASSES + 1) if diff_output_shape else TEST_NUM_CLASSES
        input = keras.Input(shape=input_dims, dtype='float32')
        x = keras.layers.Dense(output_dims)(input)
        model = keras.Model(inputs=input, outputs=x)
        return model

    def _create_test_mode_extra_layer(self):
        input = keras.Input(shape=TEST_DENSE_INPUT_DIMS, dtype='float32')
        x = keras.layers.Dense(TEST_NUM_CLASSES)(input)
        x = keras.layers.Softmax()(x)
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
    # _prune_func_smallest_weights()
    #
    def test_prune_func_smallest_weights(self):
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([1, 2, 3, 4], dtype=float), np.array([1, 1, 1, 1]), prune_percentage=0.25)
        self.assertTrue(np.array_equal([0, 1, 1, 1], actual_mask))

        # Just changed order of weights
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([3, 1, 2, 4], dtype=float), np.array([1, 1, 1, 1]), prune_percentage=0.5)
        self.assertTrue(np.array_equal([1, 0, 0, 1], actual_mask))

        # Odd number of weights
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([5, 3, 1, 2, 4], dtype=float), np.array([1, 1, 1, 1, 1]), prune_percentage=0.5)
        self.assertTrue(np.array_equal([1, 1, 0, 0, 1], actual_mask))

        # Current mask masks out one of the lowest weights
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([1, 2, 3, 4, 5], dtype=float), np.array([0, 1, 1, 1, 1]), prune_percentage=0.25)
        self.assertTrue(np.array_equal([0, 0, 1, 1, 1], actual_mask))

        # Current mask masks out one of the lowest weights
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([1, 2, 3, 4], dtype=float), np.array([0, 1, 1, 0]), prune_percentage=0.25)
        self.assertTrue(np.array_equal([0, 0, 1, 0], actual_mask))

        # Some negative and some positive weights should be masked
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([-1, 2, -3, 4], dtype=float), np.array([1, 1, 1, 1]), prune_percentage=0.5)
        self.assertTrue(np.array_equal([0, 0, 1, 1], actual_mask))

        # Many identical values but only some of them should get masked
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([1, 1, 1, 1, 2, 2], dtype=float), np.array([1, 1, 1, 1, 1, 1]), prune_percentage=0.5)
        self.assertEqual(3, np.sum(actual_mask))

        # Many identical absolute values but only some of them should get masked
        actual_mask = _prune_func_smallest_weights(np.array([]), np.array([1, -1, -1, 1, 2, -2], dtype=float), np.array([1, 1, 1, 1, 1, 1]), prune_percentage=0.5)
        self.assertEqual(3, np.sum(actual_mask))

    #
    # _prune_func_smallest_weights_global()
    #
    def test_prune_func_smallest_weights_global_negative(self):
        model = self._create_test_model()
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        # Both percentage and count are unspecified
        with self.assertRaises(ValueError) as ex:
            _ = _prune_func_smallest_weights_global(None, None, prune_percentage=None, prune_count=None)
        self.assertIn('prune_percentage', str(ex.exception))
        self.assertIn('prune_count', str(ex.exception))

        with unittest.mock.patch('logging.Logger.warning') as warning:
            _ = _prune_func_smallest_weights_global(pruner.iterate_prunables(model), None, prune_percentage=0.0, prune_count=None)
            self.assertEqual(1, warning.call_count)

        with unittest.mock.patch('logging.Logger.warning') as warning:
            _ = _prune_func_smallest_weights_global(pruner.iterate_prunables(model), None, prune_percentage=None, prune_count=0)
            self.assertEqual(1, warning.call_count)

    #
    # constructor
    #
    def test_constructor(self):
        model1 = self._create_test_model()

        # Different number of layers
        model2 = self._create_test_mode_extra_layer()
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model1)
        with self.assertRaises(ValueError) as ex:
            pruner.calc_prune_mask(model2, 0.2, 'smallest_weights')
        self.assertIn('must have the same number of layers', str(ex.exception))

        # Different shapes
        model2 = self._create_test_model_diff_shape(diff_input_shape=True)
        with self.assertRaises(ValueError) as ex:
            pruner.apply_pruning(model2)
        self.assertIn('must have the same input shape', str(ex.exception))

        model2 = self._create_test_model_diff_shape(diff_output_shape=True)
        with self.assertRaises(ValueError) as ex:
            pruner.calc_prune_mask(model2, 0.2, 'smallest_weights')
        self.assertIn('must have the same output shape', str(ex.exception))

    #
    # reset()
    #
    def test_reset(self):
        model = self._create_test_model()
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
        interesting_layer_index = 1
        interesting_weights_index = 0
        tpl = tuple([interesting_layer_index, tuple([interesting_weights_index])])

        pruner.calc_prune_mask(model, 0.2, 'smallest_weights')
        pruner.reset()

        reset_mask = np.array(pruner.prune_masks_map[tpl][interesting_weights_index])
        self.assertEqual(TEST_DENSE_WEIGHT_COUNT, np.sum(reset_mask))

    #
    # reset_masks()
    #
    def test_reset_masks(self):
        model = self._create_test_model()
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
        interesting_layer_index = 1
        interesting_weights_index = 0
        tpl = tuple([interesting_layer_index, tuple([interesting_weights_index])])

        original_mask = np.array(pruner.prune_masks_map[tpl][interesting_weights_index])
        self.assertEqual(TEST_DENSE_WEIGHT_COUNT, np.sum(original_mask))

        # Prune and make sure prune mask has changed
        pruner.calc_prune_mask(model, 0.2, 'smallest_weights')
        pruned_mask = pruner.prune_masks_map[tpl][interesting_weights_index]
        num_pruned = np.sum(pruned_mask)
        self.assertLess(num_pruned, TEST_DENSE_WEIGHT_COUNT)

        # Now reset
        pruner.reset_masks()
        reset_mask = np.array(pruner.prune_masks_map[tpl][interesting_weights_index])
        self.assertEqual(TEST_DENSE_WEIGHT_COUNT, np.sum(reset_mask))

    #
    # calc_prune_mask()
    #   'smallest_weights'
    #
    def test_smallest_weights(self):
        model = self._create_test_model()
        # First layer is the input layer; ignore it
        # Second layer is Dense layer with 2 weights. First is fully connected weights. Second is output weights.
        interesting_layer_index = 1
        interesting_layer = model.layers[interesting_layer_index]
        interesting_layer_shape = interesting_layer.weights[0].shape
        interesting_layer_weight_count = int(np.prod(interesting_layer_shape))
        interesting_key = tuple([interesting_layer_index, tuple([0])])

        dl_test_weights = np.random.choice(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES, size=TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES, replace=False)
        # Get rid of zero weights since we count those below during verification
        dl_test_weights += 1
        dl_test_weights = dl_test_weights.reshape(interesting_layer_shape)
        interesting_layer.set_weights([dl_test_weights, interesting_layer.get_weights()[1]])
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        pruner.calc_prune_mask(model, 0.5, 'smallest_weights')
        num_masked = np.sum(pruner.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.5))

        pruner.apply_pruning(model)
        actual_weights = interesting_layer.get_weights()
        actual_weights[0][actual_weights[0] == 0.0] = math.inf
        min_weight = np.min(actual_weights[0])
        self.assertGreaterEqual(min_weight, int(interesting_layer_weight_count * 0.5))

        pruner.calc_prune_mask(model, 0.2, 'smallest_weights')
        num_masked = np.sum(pruner.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.6))

        pruner.apply_pruning(model)
        actual_weights = interesting_layer.get_weights()
        actual_weights[0][actual_weights[0] == 0.0] = math.inf
        min_weight = np.min(actual_weights[0])
        self.assertGreaterEqual(min_weight, int(interesting_layer_weight_count * 0.6))

    def test_smallest_weights_2(self):
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
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        prune_rate = 0.5
        pruner.calc_prune_mask(model, prune_rate, 'smallest_weights')
        pruner.apply_pruning(model)
        actual_weights = interesting_layer.get_weights()
        min_expected_pos = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate // 2 - 1
        max_expected_neg = -TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate // 2 + 1
        unpruned_pos = np.sum(actual_weights[0] >= min_expected_pos)
        unpruned_neg = np.sum(actual_weights[0] <= max_expected_neg)
        unpruned = unpruned_pos + unpruned_neg
        self.assertIn(unpruned, [int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate), int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate) - 1])
        expected_to_be_pruned = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES - unpruned - 1
        self.assertLessEqual(abs(int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate) - expected_to_be_pruned), 1)

        # Prune again
        prune_rate2 = 0.1
        expected_to_be_pruned2 = int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate2 * (1.0 - prune_rate))
        pruner.calc_prune_mask(model, prune_rate2, 'smallest_weights')
        pruner.apply_pruning(model)
        actual_weights = interesting_layer.get_weights()
        min_expected_pos = expected_to_be_pruned2 // 2 - 1
        max_expected_neg = -expected_to_be_pruned2 // 2 + 1
        unpruned_pos = np.sum(actual_weights[0] >= min_expected_pos)
        unpruned_neg = np.sum(actual_weights[0] <= max_expected_neg)
        unpruned = unpruned_pos + unpruned_neg
        expected_unpruned = TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES - expected_to_be_pruned - expected_to_be_pruned2
        self.assertLessEqual(abs(expected_unpruned - unpruned), 1)

    def test_smallest_weights_similar_weights(self):
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
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        prune_rate = 0.5
        pruner.calc_prune_mask(model, prune_rate, 'smallest_weights')
        pruner.apply_pruning(model)
        actual_weights = interesting_layer.get_weights()
        expected = int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * prune_rate)
        actual = np.sum(actual_weights[0])
        self.assertEqual(expected, actual)

    #
    # calc_prune_mask()
    #   'smallest_weights_global'
    #
    def test_smallest_weights_global(self):
        """ Tests case where many or all weights are same value. Hence we might be tempted to mask on all of the
        smallest weights rather than honoring only up to the prune rate
        """
        random.seed(1234)
        np.random.seed(2345)
        tensorflow.set_random_seed(3456)

        model = self._create_test_dnn_model()
        interesting_layers = [model.layers[1], model.layers[4], model.layers[8]]
        interesting_weight_index = 0

        # Make sure no weights are zero so our checks below for zeroes only existing in masked weights are reliable
        weight_counts = []
        for layer in interesting_layers:
            weights = layer.get_weights()
            weights[interesting_weight_index][weights[interesting_weight_index] == 0.0] = 0.1234
            layer.set_weights(weights)
            num_weights = np.prod(weights[interesting_weight_index].shape)
            weight_counts.append(num_weights)

        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        num_pruned1 = 0
        for layer in interesting_layers:
            weights = layer.get_weights()
            num_pruned1 += np.sum(weights[interesting_weight_index] == 0.0)

        prune_rate = 0.5
        pruner.calc_prune_mask(model, prune_rate, 'smallest_weights_global')

        # calc_prune_mask() shouldn't do the actual pruning so verify that weights didn't change
        num_pruned2 = 0
        for layer in interesting_layers:
            weights = layer.get_weights()
            num_pruned2 += np.sum(weights[interesting_weight_index] == 0.0)
        self.assertEqual(num_pruned1, num_pruned2)

        pruner.apply_pruning(model)
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
        pruner.calc_prune_mask(model, prune_rate, 'smallest_weights_global')
        pruner.apply_pruning(model)

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

    #
    # calc_prune_mask()
    #   negative
    #
    def test_calc_prune_mask_negative(self):
        model = self._create_test_model()
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)
        with self.assertRaises(ValueError) as ex:
            _ = pruner.calc_prune_mask(model, 0.3, 'unknown_strategy')
        self.assertIn('smallest_weights', str(ex.exception))
        self.assertIn('smallest_weights_global', str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            _ = pruner.calc_prune_mask(model, -0.25, 'smallest_weights_global')
        self.assertIn('inclusive', str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            _ = pruner.calc_prune_mask(model, 1.1, 'smallest_weights_global')
        self.assertIn('inclusive', str(ex.exception))

    #
    # LotteryTicketPruner
    #
    def test_LotteryTicketPruner_use_case_1(self):
        model = self._create_test_model()
        starting_weights = model.get_weights()
        pruner = lottery_ticket_pruner.LotteryTicketPruner(model)

        # First layer is the input layer; ignore it
        # Second layer is Dense layer with 2 weights. First is fully connected weights. Second is output weights.
        interesting_key = tuple([1, tuple([0])])
        num_unmasked = np.sum(pruner.prune_masks_map[interesting_key][0])
        self.assertEqual(num_unmasked, TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES)

        # No pruning percent specified so no weight should change
        initial_model_weights_sum = self._summed_model_weights(model)
        pruner.apply_pruning(model)
        new_model_weights_sum = self._summed_model_weights(model)
        self.assertEqual(initial_model_weights_sum, new_model_weights_sum)

        pruner.calc_prune_mask(model, 0.5, 'random')
        num_masked = np.sum(pruner.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.5))

        pruner.calc_prune_mask(model, 0.2, 'random')
        num_masked = np.sum(pruner.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.6))

        model.set_weights(starting_weights)
        new_model_weights_sum = self._summed_model_weights(model)
        self.assertEqual(initial_model_weights_sum, new_model_weights_sum)

        pruner.apply_pruning(model)
        num_masked = np.sum(pruner.prune_masks_map[interesting_key][0] == 0)
        self.assertEqual(num_masked, int(TEST_DENSE_LAYER_INPUTS * TEST_NUM_CLASSES * 0.6))


if __name__ == '__main__':
    unittest.main()
