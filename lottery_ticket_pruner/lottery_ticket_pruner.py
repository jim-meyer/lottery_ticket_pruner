""" Copyright (C) 2020 Jim Meyer <jimm@racemed.com> """
import logging
import math
import sys

import keras
import numpy as np

logger = logging.getLogger('lottery_ticket_pruner')


def _prune_func_random(original_weights, current_weights, current_mask, prune_percentage):
    """ Randomly prunes the weights.
    This function gets called with information about the layer being pruned including the original weights, the current
    weights, the current mask. This function should not alter these.
    This function strictly calculates the updated pruning mask to be used when the `LotteryTicketPruner` instance actually
    does the pruning.
    :param original_weights: The weights as they were in the model when the `LotteryTicketPruner` instance was created.
    :param current_weights: The current weights of the model.
    :param current_mask: The current boolean mask for weight that are prunable. False means weight is pruned; True means
        it is not pruned.
    :param float prune_percentage: The percentage of all weights to be pruned. If called twice for a model with 100 weights,
        first with 0.5 then with 0.2 then the first call should prune 50 weights, the second call should prune 20 weights.
    :returns The new pruning mask. This is cumulative and hence should mask more weights than `current_mask`.
    """
    logger.debug('Randomly pruning weights of shape {}. Prune percentage={:.2f}%'.format(current_weights.shape, prune_percentage * 100.0))
    prune_count = int(np.prod(current_mask.shape) * prune_percentage)
    shape = current_mask.shape
    flat = np.ravel(current_mask)
    flat_nonzero = flat.nonzero()[0]
    mask_indices = np.random.choice(flat_nonzero, prune_count, replace=False)
    flat_mask = np.ones(flat.shape)
    np.put(flat_mask, mask_indices, 0.0)
    flat = flat * flat_mask
    return flat.reshape(shape)


def _prune_func_smallest_weights(original_weights, current_weights, current_mask, prune_percentage):
    """ Prunes the smallest magnitude (absolute value) weights.
    This function gets called with information about the layer being pruned including the original weights, the current
    weights, the current mask. This function should not alter these.
    This function strictly calculates the updated pruning mask to be used when the `LotteryTicketPruner` instance actually
    does the pruning.
    @see [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)
    :param original_weights: The weights as they were in the model when the `LotteryTicketPruner` instance was created.
    :param current_weights: The current weights of the model.
    :param current_mask: The current boolean mask for weight that are prunable. False means weight is pruned; True means
        it is not pruned.
    :param float prune_percentage: The percentage of all weights to be pruned. If called twice for a model with 100 weights,
        first with 0.5 then with 0.2 then the first call should prune 50 weights, the second call should prune 20 weights.
    :returns The new pruning mask. This is cumulative and hence should mask more weights than `current_mask`.
    """
    logger.debug('Pruning smallest weights of shape {}. Prune percentage={:.2f}%'.format(current_weights.shape, prune_percentage * 100.0))
    prune_count = int(np.prod(current_mask.shape) * prune_percentage)
    current_weights_flatten = current_weights.flatten()
    current_mask_flatten = current_mask.flatten()
    current_weights_flatten[current_mask_flatten == 0] = math.inf
    flat_abs = np.absolute(current_weights_flatten)
    flat_mins = np.argpartition(flat_abs, prune_count - 1)
    max_min = flat_abs[flat_mins[prune_count - 1]]
    possible_prune_indices = np.where(np.absolute(current_weights_flatten) <= max_min)
    prune_count = min(len(possible_prune_indices[0]), prune_count)
    prune_indices = np.random.choice(possible_prune_indices[0], prune_count, replace=False)
    new_mask_flat = current_mask_flatten
    new_mask_flat[prune_indices] = 0
    new_mask = new_mask_flat.reshape(current_mask.shape)
    return new_mask


def _prune_func_smallest_weights_global(prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
    """ Like `_prune_func_smallest_weights()` except that rather than look for smallest N weights for each layer
    we look for the smallest N weights across all layers.
    Like `_prune_func_smallest_weights()` this means the smallest magnitude weights.
    Note that in some cases more values may be pruned that requested *if* there are >1 occurrences of the `prune_count`th
    smallest value in all of the weights being pruned.
        E.g. If there are 5 occurrences of 0.1234 and 0.1234 is the `prune_count`th smallest value then 4 extra values (5 - 1 == 4)
        will be pruned. This is expected to be a rare occurrence and hence is not accounted for here.
    @see [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)
    :param iterable prunables_iterator: A iterator that returns information about what weights are prunable in the model as tuples:
        (tpl, layer, index, prune_percentage, original_weights, current_weights, current_mask)
    :param update_mask_func: A function that can be used to update the mask in the `LotteryTicketPruner` instance
    :type update_mask_func: def update_mask_func(tpl, index, new_mask)
    :param float prune_percentage:
    :returns n/a
    """
    if prune_percentage is None and prune_count is None:
        raise ValueError('Either `prune_percentage` or `prune_count` must be specified')
    weight_counts = []
    all_weights_abs = []
    current_mask_flat = []
    prunables_list = list(prunables_iterator)
    for tpl, layer, index, original_weights, current_weights, current_mask in prunables_list:
        all_weights_abs.extend(np.absolute(current_weights.flat))
        current_mask_flat.extend(current_mask.flat)
        weight_count = np.prod(current_weights.shape)
        weight_counts.append(weight_count)
    total_weight_count = np.sum(weight_counts)
    if prune_count is None:
        prune_count = int(total_weight_count * prune_percentage)
    if prune_count == 0:
        logger.warning('{} called with parameters indicating no pruning should be done'.format(sys._getframe().f_code.co_name))
        return

    logger.info('Pruning {} of {} total weights ({:.2f}%)'.format(prune_count, total_weight_count, prune_count / total_weight_count * 100))
    current_mask_flat = np.array(current_mask_flat)
    all_weights_abs = np.array(all_weights_abs)
    all_weights_abs[current_mask_flat == False] = math.inf  # noqa - flake8 complains about this but "current_mask_flat is False" does *not* work
    flat_mins = np.argpartition(all_weights_abs, prune_count - 1)
    max_min = all_weights_abs[flat_mins[prune_count - 1]]

    for tpl, layer, index, original_weights, current_weights, current_mask in prunables_list:
        new_mask = np.absolute(current_weights) > max_min
        pruned_count = np.sum(new_mask == 0)
        weight_count = np.prod(current_weights.shape)
        logger.debug('Globally pruning {} of {} ({:.2f}%) smallest weights of layer {}/{}'.format(pruned_count, weight_count, pruned_count / weight_count * 100, layer.name, index))
        update_mask_func(tpl, index, new_mask)


# def prune_func_same_sign(prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
#     """ Prunes weights across all layers that don't have the same sign as the original weights.
#     we look for the smallest N weights across all layers.
#     @see[Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)
#     :param iterable prunables_iterator: A iterator that returns information about what weights are prunable in the model as tuples:
#         (tpl, index, prune_percentage, original_weights, current_weights, current_mask)
#     :param update_mask_func: A function that can be used to update the mask in the `LotteryTicketPruner` instance
#     :type update_mask_func: def update_mask_func(tpl, index, new_mask)
#     :param float prune_percentage:
#     :returns n/a
#     """
#     if prune_percentage is None and prune_count is None:
#         raise ValueError('Either `prune_percentage` or `prune_count` must be specified')
#     raise NotImplementedError('Not implemented yet')


class LotteryTicketPruner(object):
    """
    This class prunes weights from a model and keeps internal state to track which weights have been pruned.
    Inspired from https://arxiv.org/pdf/1803.03635.pdf, "THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS"
    """
    def __init__(self, model, original_model=None):
        """
        :param model: The model being trained and that will be pruned.
        :param original_model: (optional) The model containing the original weights to be used by the pruning logic to
            determine which weights of the model being trained are to be pruned.
            If `None` then the original weights are taken from `model`.
        """
        self.model = model

        if original_model is None:
            original_model = model
        else:
            if len(self.model.layers) != len(original_model.layers):
                raise ValueError('`model` and `original_model` must have the same number of layers ({} vs {}'.format(
                    len(self.model.layers), len(original_model.layers)))
            for layer, original_layer in zip(self.model.layers, original_model.layers):
                if (hasattr(layer, 'input_shape') or hasattr(original_layer, 'input_shape')) and layer.input_shape != original_layer.input_shape:
                    raise ValueError(
                        'All layers in `model` and `original_model` must have the same input shape ({}/{} vs {}/{})'.format(
                            layer.name, layer.input_shape, original_layer.name, original_layer.input_shape))

                if (hasattr(layer, 'output_shape') or hasattr(original_layer, 'output_shape')) and layer.output_shape != original_layer.output_shape:
                    raise ValueError(
                        'All layers in `model` and `original_model` must have the same output shape ({}/{} vs {}/{})'.format(
                            layer.name, layer.output_shape, original_layer.name, original_layer.output_shape))

        # Now determine which weights of which layers are prunable
        layer_index = 0
        # An array of (layer index, [weight indices (ints)]) tuples of weights that are prunable.
        self.prunable_tuples = []
        # Dicts of (layer, [weight indices (ints)]) tuples whose value are arrays of masks, arrays of weights
        self.prune_masks_map = {}
        self.prunable_weights_map = {}
        self._original_weights_map = {}
        for layer in original_model.layers:
            layer_weights = layer.get_weights()
            prune_masks = [None] * len(layer_weights)
            original_layer_weights = [None] * len(layer_weights)
            weights_indices = set()
            i = 0
            for weights in layer_weights:
                if self._prunable(layer, weights):
                    weights_indices.add(i)
                    prune_masks[i] = np.ones(weights.shape)
                    original_layer_weights[i] = weights
                i += 1
            if len(weights_indices) > 0:
                tpl = (layer_index, tuple(weights_indices))
                self.prunable_tuples.append(tpl)
                self.prune_masks_map[tpl] = prune_masks
                self._original_weights_map[tpl] = original_layer_weights
            layer_index += 1
        self.cumulative_pruning_rate = 0.0

    def _prunable(self, layer, weights):
        return isinstance(layer, (keras.layers.Conv1D,
                                  keras.layers.SeparableConv1D,
                                  keras.layers.Conv2D, keras.layers.Conv2DTranspose, keras.layers.Convolution2D,
                                  keras.layers.Convolution2DTranspose,
                                  keras.layers.DepthwiseConv2D,
                                  keras.layers.SeparableConv2D,
                                  keras.layers.Conv3D, keras.layers.Convolution3D,
                                  keras.layers.Dense,
                                  )) and len(weights.shape) > 1

    def reset(self):
        """ Resets instance to the state it was initially constructed in.
            Model's original weights are restored.
            Pruned weight masks are reset.
        """
        self.reset_masks()

    def reset_masks(self):
        """ Resets the instance's pruning mask to it's initial state (nothing gets pruned). """
        for tpl, masks in self.prune_masks_map.items():
            for index in tpl[1]:
                assert masks[index] is not None
                masks[index].fill(True)

    def _update_mask(self, tpl, index, new_mask):
        assert self.prune_masks_map[tpl][index] is not None
        self.prune_masks_map[tpl][index] = new_mask

    def iterate_prunables(self):
        """ Returns iterator over all prunable weights in all layers of the model.
        returns: tuple of (tpl<layer index, index of weights in layer>, layer, index of these weights in layer's weights array,
                            prune percentage, original weights, current weights, prune mask)
        """
        for tpl in self.prunable_tuples:
            layer_index = tpl[0]
            indices = tpl[1]
            layer = self.model.layers[layer_index]
            current_weights = layer.get_weights()
            for index in indices:
                mask = self.prune_masks_map[tpl][index]
                if mask is not None:    # TODO - why would mask ever be none here?
                    original_weights = self._original_weights_map[tpl][index]
                    yield tpl, layer, index, original_weights, current_weights[index], mask

    def prune_weights(self, prune_percentage, prune_strategy):
        """ Prunes the specified percentage of the remaining unpruned weights from the model.
        This updates the model's weights such that they are now pruned.
        This also updates the internal pruned weight masks managed by this instance. @see `apply_pruning()`
        :param float prune_percentage: The percentage *of remaining unpruned weights* to be pruned.
            Note that these pruning percentages are cumulative within an instance of this class.
            E.g. Calling `prune_weights()` twice, once with 0.5 and again with 0.2 will result in 60% pruning.
                (0.5 + (1.0 - 0.5) * 0.2) = 0.60
        :param string prune_strategy: A string indicating how the pruning should be done.
            'random': Pruning is done randomly across all prunable layers.
            'smallest_weights': The smallest weights at each prunable layer are pruned. Each prunable layer has the
                specified percentage pruned from the layer's weights.
            'smallest_weights_global': The smallest weights across all prunable layers are pruned. Some layers may have
                substantially more or less weights than `prune_percentage` pruned from them. But overall, across all
                prunable layers, `prune_percentage` weights will be pruned.
        """
        if not (0.0 < prune_percentage < 1.0):
            raise ValueError('"prune_percentage" must be between 0.0 and 1.0 inclusive but it was {}'.format(prune_percentage))

        # Convert from percentage of remaining to percentage overall since the latter is easier for pruning functions to use
        actual_prune_percentage = (1.0 - self.cumulative_pruning_rate) * prune_percentage
        self.cumulative_pruning_rate += actual_prune_percentage

        local_prune_strats = {'random': _prune_func_random, 'smallest_weights': _prune_func_smallest_weights}
        global_prune_strats = {'smallest_weights_global': _prune_func_smallest_weights_global}
        if prune_strategy in local_prune_strats:
            local_prune_func = local_prune_strats[prune_strategy]
            for tpl, layer, index, original_weights, current_weights, mask in self.iterate_prunables():
                new_mask = local_prune_func(original_weights, current_weights, mask, actual_prune_percentage)
                self.prune_masks_map[tpl][index] = new_mask
        elif prune_strategy in global_prune_strats:
            global_prune_func = global_prune_strats[prune_strategy]
            global_prune_func(self.iterate_prunables(), self._update_mask, prune_percentage=actual_prune_percentage)
        else:
            all_keys = set(local_prune_strats.keys()).union(set(global_prune_strats.keys()))
            raise ValueError('"prune_strategy" must be one of {}'.format(all_keys))

        # Now prune the weights from the model's layers
        self.apply_pruning()

    def apply_pruning(self):
        """ Applies the existing pruning masks to the model's weights """
        for tpl in self.prunable_tuples:
            layer_index = tpl[0]
            weight_indices = tpl[1]
            layer = self.model.layers[layer_index]
            prune_masks = self.prune_masks_map[tpl]

            layer_weights = layer.get_weights()
            for index in weight_indices:
                assert prune_masks[index] is not None
                layer_weights[index] *= prune_masks[index]
            layer.set_weights(layer_weights)
