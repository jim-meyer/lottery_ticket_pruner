""" Copyright 2020 Jim Meyer """
import copy
import logging
import math

import keras
import numpy as np

logger = logging.get_logger('lottery_ticket_pruner')


def prune_func_docs(original_weights, current_weights, current_mask, prune_percentage=None, prune_count=None):
    """ Calculates the pruning mask for a given layer.
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
        Mutually exclusive with `prune_count`.
    :param int prune_count: The number weights to be pruned. Mutually exclusive with `prune_percentage`.
    :raises ValueError if both, or neither, of `prune_percentage` and `prune_count` are specified.
    """
    raise NotImplementedError('This function exists just for documentation purposes')


def prune_func_global_docs(prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
    """ *_global() prune functions get information about all the prubable weights in the entire model. This allows
    the global pruning function to operate across "globally" across all layers if it so chooses.
    :param iterable prunables_iterator: A iterator that returns information about what weights are prunable in the model as tuples:
        (tpl, index, prune_percentage, original_weights, current_weights, current_mask)
    :param update_mask_func: A function that can be used to update the mask in the `LotteryTicketPruner` instance
    :type update_mask_func: def update_mask_func(tpl, index, new_mask)
    :param float prune_percentage: The percentage of all weights to be pruned. If called twice for a model with 100 weights,
        first with 0.5 then with 0.2 then the first call should prune 50 weights, the second call should prune 20 weights.
        Mutually exclusive with `prune_count`.
    :param int prune_count: The number weights to be pruned. Mutually exclusive with `prune_percentage`.
    :returns n/a
    :raises ValueError if both, or neither, of `prune_percentage` and `prune_count` are specified.
    """
    raise NotImplementedError('This function exists just for documentation purposes')


def prune_func_random(original_weights, current_weights, current_mask, prune_percentage=None, prune_count=None):
    if prune_percentage is None and prune_count is None:
        raise ValueError('Either `prune_percentage` or `prune_count` must be specified')
    if prune_count is None:
        prune_count = int(np.prod(current_mask.shape) * prune_percentage)
    if prune_count == 0:
        return current_mask

    shape = current_mask.shape
    flat = np.ravel(current_mask)
    flat_nonzero = flat.nonzero()[0]
    mask_indices = np.random.choice(flat_nonzero, prune_count, replace=False)
    flat_mask = np.ones(flat.shape)
    np.put(flat_mask, mask_indices, 0.0)
    flat = flat * flat_mask
    return flat.reshape(shape)


def prune_func_smallest_weights(original_weights, current_weights, current_mask, prune_percentage=None, prune_count=None):
    if prune_percentage is None and prune_count is None:
        raise ValueError('Either `prune_percentage` or `prune_count` must be specified')
    if prune_count is None:
        prune_count = int(np.prod(current_mask.shape) * prune_percentage)
    if prune_count == 0:
        return current_mask

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


def prune_func_smallest_weights_global(prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
    """ Like `prune_func_smallest_weights()` except that rather than look for smallest N weights for each layer
    we look for the smallest N weights across all layers.
    Note that in some cases more values may be pruned that requested *if* there are >1 occurrences of the `prune_count`th
    smallest value in all of the weights being pruned.
        E.g. If there are 5 occurrences of 0.1234 and 0.1234 is the `prune_count`th smallest value then 4 extra values (5 - 1 == 4)
        will be pruned. This is expected to be a rare occurrence and hence is not accounted for here.
    :param float prune_percentage:
    :param iterable prunables_iterator: A iterator that returns information about what weights are prunable in the model as tuples:
        (tpl, index, prune_percentage, original_weights, current_weights, current_mask)
    :param update_mask_func: A function that can be used to update the mask in the `LotteryTicketPruner` instance
    :type update_mask_func: def update_mask_func(tpl, index, new_mask)
    :returns n/a
    """
    if prune_percentage is None and prune_count is None:
        raise ValueError('Either `prune_percentage` or `prune_count` must be specified')
    weight_counts = []
    all_weights_abs = []
    current_mask_flat = []
    prunables_list = list(prunables_iterator)
    for tpl, index, original_weights, current_weights, current_mask in prunables_list:
        all_weights_abs.extend(np.absolute(current_weights.flat))
        current_mask_flat.extend(current_mask.flat)
        weight_count = np.prod(current_weights.shape)
        weight_counts.append(weight_count)
    total_weight_count = np.sum(weight_counts)
    if prune_count is None:
        prune_count = int(total_weight_count * prune_percentage)
    if prune_count == 0:
        return

    logger.info('Pruning {} of {} total weights ({:.2f}%)'.format(prune_count, total_weight_count, prune_count / total_weight_count * 100))
    current_mask_flat = np.array(current_mask_flat)
    all_weights_abs = np.array(all_weights_abs)
    all_weights_abs[current_mask_flat == False] = math.inf  # noqa - flake8 complains about this but "current_mask_flat is False" does *not* work
    flat_mins = np.argpartition(all_weights_abs, prune_count - 1)
    max_min = all_weights_abs[flat_mins[prune_count - 1]]

    for tpl, index, original_weights, current_weights, current_mask in prunables_list:
        new_mask = np.absolute(current_weights) > max_min
        pruned_count = np.sum(new_mask == 0)
        weight_count = np.prod(current_weights.shape)
        logger.info('Pruning {} of {} weights ({:.2f}%) of layer {}/{}'.format(pruned_count, weight_count, pruned_count / weight_count * 100, tpl[0].name, tpl[1]))
        # print('Global pruning @{:.2f}%: Current mask count = {}. New mask count = {} out of {}'.format(prune_count / total_weight_count * 100.0,
        #                                                                                                np.sum(current_mask == False),
        #                                                                                                np.sum(new_mask == False),
        #                                                                                                np.prod(current_mask.shape)))
        update_mask_func(tpl, index, new_mask)


# def prune_func_same_sign(prune_percentage, original_weights, current_weights, current_mask):
#     prune_count = int(np.sum(current_mask) * prune_percentage)
#     original_flat = original_weights.flatten()
#     flat = current_weights.flatten()
#     flat[flat == 0.0] = math.inf
#
#     flat_mins = np.argpartition(np.absolute(flat), prune_count - 1)
#     max_min = flat[flat_mins[prune_count - 1]]
#
#     prune_mask = current_weights > max_min
#     new_mask = current_mask * prune_mask
#     return new_mask


class LotteryTicketPruner(object):
    """
    This class prunes weights from a model and keeps internal state to track which weights have been pruned.
    Inspired from https://arxiv.org/pdf/1803.03635.pdf, "THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS"
    """
    def __init__(self, model):
        self.model = model

        self._original_weights = []
        for layer in model.layers:
            layer_weights = layer.get_weights()
            self._original_weights.append(copy.deepcopy(layer_weights))

        # An array of (layer, [weight indices (ints)]) tuples of weights that are prunable.
        self.prunable_tuples = []
        # Dicts of (layer, [weight indices (ints)]) tuples whose value are arrays of masks, arrays of weights
        self.prune_masks_map = {}
        self.prunable_weights_map = {}
        self.original_weights_map = {}
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            prune_masks = [None] * len(layer_weights)
            weights_indices = set()
            i = 0
            for weights in layer_weights:
                if self._prunable(layer, weights):
                    weights_indices.add(i)
                    prune_masks[i] = np.ones(weights.shape)
                i += 1
            tpl = (layer, tuple(weights_indices))
            self.prunable_tuples.append(tpl)
            self.prune_masks_map[tpl] = prune_masks
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
        self.restore_initial_weights()
        self.reset_mask()

    def reset_mask(self):
        """ Resets the instance's pruning mask to it's initial state (nothing gets pruned). """
        for tpl, masks in self.prune_masks_map.items():
            for index in tpl[1]:
                assert masks[index] is not None
                masks[index].fill(True)

    def restore_initial_weights(self):
        """ Resets model to use the weights that this instance was initially constructed with. """
        for layer, w in zip(self.model.layers, self._original_weights):
            layer.set_weights(w)

    def update_mask(self, tpl, index, new_mask):
        assert self.prune_masks_map[tpl][index] is not None
        self.prune_masks_map[tpl][index] = new_mask

    def iterate_prunables(self):
        """ Returns iterator over all prunable weights in all layers of the model.
        returns: tuple of (tpl<layer, index of weights in layer>, index, prune percentage, original weights, current weights, prune mask)
        """
        for tpl in self.prunable_tuples:
            layer = tpl[0]
            indices = tpl[1]
            current_weights = layer.get_weights()
            for index in indices:
                mask = self.prune_masks_map[tpl][index]
                if mask is not None:
                    yield tpl, index, [], current_weights[index], mask

    def prune_weights(self, prune_percentage, local_prune_func=None, global_prune_func=None):
        """ Prunes the specified percentage of the remaining unpruned weights from the model.
        This updates the model's weights such that they are now pruned.
        This also updates the internal pruned weight masks managed by this instance.
        :param float prune_percentage: The percentage *of remaining unpruned weights* to be pruned.
            Note that these pruning percentages are cumulative within an instance of this class.
            E.g. Calling `prune_weights()` twice, once with 0.5 and again with 0.2 will result in 60% pruning.
                (0.5 + (1.0 - 0.5) * 0.2) = 0.60
        :param local_prune_func: The function to call when deciding what gets pruned.
                This function can only do local pruning on one set of weights at a time (vs global pruning
                which prunes across the entirety of all prunable weights in a model at a time)
        :type function:
                def local_prune_func(original_weights, current_weights, current_mask, prune_percentage=None, prune_count=None):
                    return <pruned weight mask: 0 -> weight will be pruned; 1 -> weight will not be pruned>
        """
        if not (0.0 < prune_percentage < 1.0):
            raise ValueError('"prune_percentage" must be between 0.0 and 1.0 inclusive but it was {}'.format(prune_percentage))

        # Convert from percentage of remaining to percentage overall since the latter is easier for pruning functions to use
        actual_prune_percentage = (1.0 - self.cumulative_pruning_rate) * prune_percentage
        self.cumulative_pruning_rate += actual_prune_percentage

        if local_prune_func is not None:
            for tpl, index, original_weights, current_weights, mask in self.iterate_prunables():
                new_mask = local_prune_func(original_weights, current_weights, mask, prune_percentage=actual_prune_percentage)
                self.prune_masks_map[tpl][index] = new_mask
        elif global_prune_func is not None:
            def proxy(real_prune_func, prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
                prunables_iterator = list(prunables_iterator)
                for tpl, index, original_weights, current_weights, current_mask in prunables_iterator:
                    logger.info('Global pruning: tpl={}, index={}, weights.shape={}'.format(tpl, index, current_weights.shape))
                real_prune_func(prunables_iterator, update_mask_func, prune_percentage=prune_percentage, prune_count=prune_count)

            # proxy(global_prune_func, self.iterate_prunables(), self.update_mask, prune_percentage=actual_prune_percentage)
            global_prune_func(self.iterate_prunables(), self.update_mask, prune_percentage=actual_prune_percentage)
        else:
            raise ValueError('Either `local_prune_func` or `global_prune_func` must be specified')

        # Now prune the weights from the model's layers
        self.apply_pruning()

    def apply_pruning(self):
        """ Applies the existing pruning masks to the model's weights """
        for tpl in self.prunable_tuples:
            layer = tpl[0]
            weight_indices = tpl[1]
            prune_masks = self.prune_masks_map[tpl]

            layer_weights = layer.get_weights()
            for index in weight_indices:
                assert prune_masks[index] is not None
                layer_weights[index] *= prune_masks[index]
            layer.set_weights(layer_weights)
