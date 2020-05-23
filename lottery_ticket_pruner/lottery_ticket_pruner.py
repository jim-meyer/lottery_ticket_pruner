import logging
import math
import sys

import keras
import numpy as np

logger = logging.getLogger('lottery_ticket_pruner')


def _prune_func_random(initial_weights, pretrained_weights, current_weights, current_mask, prune_percentage):
    """ Randomly prunes the weights.
    This function gets called with information about the layer being pruned including the initial weights, the current
    weights, the current mask. This function should not alter these.
    This function strictly calculates the updated pruning mask to be used when the `LotteryTicketPruner` instance actually
    does the pruning.
    :param initial_weights: The weights as they were in the untrained model when the `LotteryTicketPruner` instance was created.
        (Unused by this pruning strategy)
    :param pretrained_weights: The weights from a fully trained model. (Unused by this pruning strategy)
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


def _prune_func_smallest_weights(initial_weights, pretrained_weights, current_weights, current_mask, prune_percentage):
    """ Prunes the smallest magnitude (absolute value) weights.
    This function gets called with information about the layer being pruned including the initial weights, the current
    weights, the current mask. This function should not alter these.
    This function strictly calculates the updated pruning mask to be used when the `LotteryTicketPruner` instance actually
    does the pruning.
    @see [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)
    :param initial_weights: The weights as they were in the untrained model when the `LotteryTicketPruner` instance was created.
        (Unused by this pruning strategy)
    :param pretrained_weights: The weights from a fully trained model. (Unused by this pruning strategy)
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
        (tpl, layer, index, prune_percentage, initial_weights, current_weights, current_mask)
    :param update_mask_func: A function that can be used to update the mask in the `LotteryTicketPruner` instance
    :type update_mask_func: def update_mask_func(tpl, index, new_mask)
    :param float prune_percentage: The percentage of all weights to be pruned. If called twice for a model with 100 weights,
        first with 0.5 then with 0.2 then the first call should prune 50 weights, the second call should prune 20 weights.
    :param int prune_count: The number of additional weights to be pruned. If called twice for a model with 100 weights,
        first with 50 then with 20 then the first call should prune 50 weights, the second call should prune 20 additional weights.
    :returns n/a
    """
    if prune_percentage is None and prune_count is None:
        raise ValueError('Either `prune_percentage` or `prune_count` must be specified')

    weight_counts = []
    all_weights_abs = []
    current_mask_flat = []
    prunables_list = list(prunables_iterator)
    for tpl, layer, index, initial_weights, _, current_weights, current_mask in prunables_list:
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

    for tpl, layer, index, initial_weights, _, current_weights, current_mask in prunables_list:
        # Just in case `current_weights` isn't in a pruned state
        current_weights *= current_mask
        new_mask = np.absolute(current_weights) > max_min
        pruned_count = np.sum(new_mask == 0)
        weight_count = np.prod(current_weights.shape)
        logger.debug('Globally pruning {} of {} ({:.2f}%) smallest weights of layer {}/{}'.format(pruned_count, weight_count, pruned_count / weight_count * 100, layer.name, index))
        update_mask_func(tpl, index, new_mask)


def _prune_func_large_final(prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
    """ Prunes weights by keeping those with the largest magnitude from the fully trained model.
    This is 'large_final' as defined in https://arxiv.org/pdf/1905.01067.pdf
    :param iterable prunables_iterator: A iterator that returns information about what weights are prunable in the model as tuples:
        (tpl, index, prune_percentage, initial_weights, current_weights, current_mask)
    :param update_mask_func: A function that can be used to update the mask in the `LotteryTicketPruner` instance
    :type update_mask_func: def update_mask_func(tpl, index, new_mask)
    :param float prune_percentage: The percentage of all weights to be pruned. If called twice for a model with 100 weights,
        first with 0.5 then with 0.2 then the first call should prune 50 weights, the second call should prune 20 weights.
    :param int prune_count: The number of additional weights to be pruned. If called twice for a model with 100 weights,
        first with 50 then with 20 then the first call should prune 50 weights, the second call should prune 20 additional weights.
    :returns n/a
    """
    if prune_percentage is None and prune_count is None:
        raise ValueError('Either `prune_percentage` or `prune_count` must be specified')

    use_prune_percentage = prune_count is None

    for tpl, layer, index, initial_weights, pretrained_weights, current_weights, current_mask in prunables_iterator:
        if pretrained_weights is None:
            raise ValueError('"large_final" pruning strategy requires that `LotteryTicketPruner.pretrained_weights()` be called with a pre-trained model')

        weight_count = np.prod(pretrained_weights.shape)
        if use_prune_percentage:
            prune_count = int(weight_count * prune_percentage)
        if prune_count == 0:
            logger.warning('{} called with parameters indicating no pruning should be done'.format(sys._getframe().f_code.co_name))
            return
        # The logic below assumes `prune_count` is the total number of weights to be pruned
        prune_count += np.sum(current_mask == 0)
        assert prune_count < weight_count

        # Just in case `current_weights` isn't in a pruned state
        pretrained_weights *= current_mask
        trained_weights_flat = pretrained_weights.flat
        trained_weights_flat = np.abs(trained_weights_flat)
        flat_mins = np.argpartition(trained_weights_flat, prune_count - 1)
        max_min = trained_weights_flat[flat_mins[prune_count - 1]]
        max_min = np.abs(max_min)

        new_mask = np.absolute(pretrained_weights) > max_min
        pruned_count = np.sum(new_mask == 0)
        logger.debug('Pruning {} of {} ({:.2f}%) using "large_final" for layer {}/{}'.format(pruned_count, weight_count, pruned_count / weight_count * 100, layer.name, index))
        update_mask_func(tpl, index, new_mask)


# def prune_func_same_sign(prunables_iterator, update_mask_func, prune_percentage=None, prune_count=None):
#     """ Prunes weights across all layers that don't have the same sign as the initial weights.
#     we look for the smallest N weights across all layers.
#     @see[Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)
#     :param iterable prunables_iterator: A iterator that returns information about what weights are prunable in the model as tuples:
#         (tpl, index, prune_percentage, initial_weights, current_weights, current_mask)
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
    def __init__(self, initial_model):
        """
        :param initial_model: The model containing the initial weights to be used by the pruning logic to determine
            which weights of the model being trained are to be pruned.
            Some pruning strategies require access to the initial weights of the model to determine the pruning mask.
        """

        # Now determine which weights of which layers are prunable
        layer_index = 0
        # An array of (layer index, [weight indices (ints)]) tuples of weights that are prunable.
        self.prunable_tuples = []
        # Dicts of (layer, [weight indices (ints)]) tuples whose value are arrays of masks, arrays of weights
        self.prune_masks_map = {}
        self._initial_weights_map = {}
        self.layer_input_shapes = []
        self.layer_output_shapes = []
        self.model_weight_shapes = []
        self._pretrained_weights = None
        for layer in initial_model.layers:
            layer_weights = layer.get_weights()
            self.model_weight_shapes += [w.shape for w in layer_weights]
            weight_shapes = [None] * len(layer_weights)
            prune_masks = [None] * len(layer_weights)
            initial_layer_weights = [None] * len(layer_weights)
            weights_indices = set()
            i = 0
            for weights in layer_weights:
                weight_shapes[i] = weights.shape
                if self._prunable(layer, weights):
                    weights_indices.add(i)
                    prune_masks[i] = np.ones(weights.shape)
                    initial_layer_weights[i] = weights
                i += 1
            if len(weights_indices) > 0:
                tpl = (layer_index, tuple(weights_indices))
                self.prunable_tuples.append(tpl)
                self.prune_masks_map[tpl] = prune_masks
                self._initial_weights_map[tpl] = initial_layer_weights
            self.layer_input_shapes.append(getattr(layer, 'input_shape', None))
            self.layer_output_shapes.append(getattr(layer, 'output_shape', None))
            layer_index += 1
        self.cumulative_pruning_rate = 0.0

    def _verify_compatible_model(self, model):
        if len(model.layers) != len(self.layer_input_shapes):
            raise ValueError('`model` must have the same number of layers as the initial model used to create this instance ({} vs {}'.format(
                len(model.layers), len(self.layer_input_shapes)))
        for layer, input_shape, output_shape in zip(model.layers, self.layer_input_shapes, self.layer_output_shapes):
            if getattr(layer, 'input_shape', None) != input_shape:
                raise ValueError(
                    'All layers in `model` and `initial_model` must have the same input shape for layer {} ({} vs {})'.format(
                        layer.name, getattr(layer, 'input_shape', None), input_shape))

            if getattr(layer, 'output_shape', None) != output_shape:
                raise ValueError(
                    'All layers in `model` and `initial_model` must have the same output shape {} ({} vs {})'.format(
                        layer.name, getattr(layer, 'output_shape', None), output_shape))

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

    def apply_dwr(self, model):
        """ Applies Dynamic Weight Rescaling (DWR) to the unpruned weights in the model.
        See section 5.2, "Dynamic Weight Rescaling" of https://arxiv.org/pdf/1905.01067.pdf.
        A quote from that paper describes it best:
            "For each training iteration and for each layer, we multiply the underlying weights by the ratio of the total
            number of weights in the layer over the number of ones in the corresponding mask."
        The typical use of this is to call `apply_dwr()` right after `apply_pruning()` per the above paper.
        This is left as a separate step, and not done implicitly by `apply_pruning()`, since whereas `apply_pruning()`
        can be called repeatedly with effect beyond the first call, `apply_dwr()` should be called exactly once after
        `apply_pruning()` if DWR is desired.
        :param model: The model whose unpruned weights should be rescaled.
        """
        for tpl, layer, index, _, _, current_weights, mask in self.iterate_prunables(model):
            num_weights = np.prod(current_weights.shape)
            num_ones = np.sum(mask == 1)
            scale_factor = num_weights / num_ones
            current_weights *= scale_factor
            layer_weights = layer.get_weights()
            layer_weights[index] = current_weights
            layer.set_weights(layer_weights)

    def set_pretrained_weights(self, pretrained_model):
        """ Some pruning strategires require acccess to a trained model's weights. This method lets the pruner
        know what the weights are for a trained model.
        `pretrained_model` should be trained starting from the initial weights of the model that was used to construct
        this instance or else Lottery Ticket pruning won't work well.
        For example the 'large_final' strategy. See `LotteryTicketPruner.calc_prune_mask()`.
        Example usage:
            model = <create model with random initial weights>
            # Save the initial weights of the model so we can start pruning training from them later
            initial_weights = model.get_weights()
            # Initialize pruner so it knows the starting initial (random) weights
            pruner = LotteryTicketPruner(model)
            ...
            # Train the model
            model.fit(X, y)
            ...
            pruner.set_pretrained_weights(model)
            ...
            # Revert model so it has random initial weights
            model.set_weights(initial_weights)
            # Now train the model using pruning
            pruner.calc_prune_mask(model, 0.5, 'large_final')
            untrained_loss, untrained_accuracy = model.evaluate(x_test, y_test)
            model.fit(X, y, callbacks=[PrunerCallback(pruner)])
            trained_loss, trained_accuracy = model.evaluate(x_test, y_test)
        """
        self._verify_compatible_model(pretrained_model)

        pretrained_weights_map = {}
        for tpl in self.prunable_tuples:
            layer_index = tpl[0]
            indices = tpl[1]
            layer = pretrained_model.layers[layer_index]
            layer_weights = layer.get_weights()
            retained_weights = [None] * len(layer_weights)
            for index in indices:
                retained_weights[index] = layer_weights[index]
            pretrained_weights_map[tpl] = retained_weights
        self._pretrained_weights = pretrained_weights_map

    def reset_masks(self):
        """ Resets the instance's pruning mask to it's initial state (so nothing gets pruned). """
        for tpl, masks in self.prune_masks_map.items():
            for index in tpl[1]:
                assert masks[index] is not None
                masks[index].fill(True)

    def _update_mask(self, tpl, index, new_mask):
        assert self.prune_masks_map[tpl][index] is not None
        self.prune_masks_map[tpl][index] = new_mask

    def iterate_prunables(self, model):
        """ Returns iterator over all prunable weights in all layers of the model.
        returns: tuple of (tpl<layer index, index of weights in layer>, layer, index of these weights in layer's weights array,
                            prune percentage, initial weights, current weights, prune mask)
        """
        self._verify_compatible_model(model)

        for tpl in self.prunable_tuples:
            layer_index = tpl[0]
            indices = tpl[1]
            layer = model.layers[layer_index]
            current_weights = layer.get_weights()
            for index in indices:
                mask = self.prune_masks_map[tpl][index]
                if mask is not None:    # TODO - why would mask ever be none here?
                    initial_weights = self._initial_weights_map[tpl][index]
                    pretrained_weights = self._pretrained_weights[tpl][index] if self._pretrained_weights is not None else None
                    yield tpl, layer, index, initial_weights, pretrained_weights, current_weights[index], mask

    def calc_prune_mask(self, model, prune_percentage, prune_strategy):
        """ Prunes the specified percentage of the remaining unpruned weights from the model.
        This updates the model's weights such that they are now pruned.
        This also updates the internal pruned weight masks managed by this instance. @see `apply_pruning()`
        :param model: The model that contains the current weights to be used to calculate the pruning mask.
            Typically this is the model being trained while being pruned every epoch.
        :param float prune_percentage: The percentage *of remaining unpruned weights* to be pruned.
            Note that these pruning percentages are cumulative within an instance of this class.
            E.g. Calling `calc_prune_mask()` twice, once with 0.5 and again with 0.2 will result in 60% pruning.
                (0.5 + (1.0 - 0.5) * 0.2) = 0.60
        :param string prune_strategy: A string indicating how the pruning should be done.
            'random': Pruning is done randomly across all prunable layers.
            'smallest_weights': The smallest weights at each prunable layer are pruned. Each prunable layer has the
                specified percentage pruned from the layer's weights.
            'smallest_weights_global': The smallest weights across all prunable layers are pruned. Some layers may have
                substantially more or less weights than `prune_percentage` pruned from them. But overall, across all
                prunable layers, `prune_percentage` weights will be pruned.
            'large_final': Keeps the weights that have the largest magnitude from the previously trained model.
                This is 'large_final' as defined in https://arxiv.org/pdf/1905.01067.pdf
            'large_final': Keeps the weights that have the largest magnitude from the previously trained model.
                This is 'large_final' as defined in https://arxiv.org/pdf/1905.01067.pdf
            'large_final_same_sign': TODO - "same sign" logic needs to be applied once to the model prior to training, not during pruning.
        """
        if not (0.0 < prune_percentage < 1.0):
            raise ValueError('"prune_percentage" must be between 0.0 and 1.0 exclusive but it was {}'.format(prune_percentage))
        self._verify_compatible_model(model)

        # Convert from percentage of remaining to percentage overall since the latter is easier for pruning functions to use
        actual_prune_percentage = (1.0 - self.cumulative_pruning_rate) * prune_percentage
        self.cumulative_pruning_rate += actual_prune_percentage

        local_prune_strats = {'random': _prune_func_random, 'smallest_weights': _prune_func_smallest_weights}
        global_prune_strats = {'smallest_weights_global': _prune_func_smallest_weights_global,
                               'large_final': _prune_func_large_final}
        if prune_strategy in local_prune_strats:
            local_prune_func = local_prune_strats[prune_strategy]
            for tpl, layer, index, initial_weights, pretrained_weights, current_weights, mask in self.iterate_prunables(model):
                new_mask = local_prune_func(initial_weights, pretrained_weights, current_weights, mask, actual_prune_percentage)
                self.prune_masks_map[tpl][index] = new_mask
        elif prune_strategy in global_prune_strats:
            global_prune_func = global_prune_strats[prune_strategy]
            global_prune_func(self.iterate_prunables(model), self._update_mask, prune_percentage=actual_prune_percentage)
        else:
            all_keys = set(local_prune_strats.keys()).union(set(global_prune_strats.keys()))
            raise ValueError('"prune_strategy" must be one of {}'.format(all_keys))

    def apply_pruning(self, model):
        """ Applies the existing pruning masks to the model's weights """
        self._verify_compatible_model(model)

        for tpl in self.prunable_tuples:
            layer_index = tpl[0]
            weight_indices = tpl[1]
            layer = model.layers[layer_index]
            prune_masks = self.prune_masks_map[tpl]

            layer_weights = layer.get_weights()
            for index in weight_indices:
                assert prune_masks[index] is not None
                layer_weights[index] *= prune_masks[index]
            layer.set_weights(layer_weights)
