import tensorflow.keras as keras
from typing import List
import lottery_ticket_pruner.lottery_ticket_pruner as ltp

class PrunerCallback(keras.callbacks.Callback):
    """  """
    def __init__(self, pruner, use_dwr=False, prune_every_batch_iteration=False, iterative_model_non_zero_dense_weights_after_pruning=None,
                 iterative_model_non_zero_convolutional_weights_after_pruning=None, recalculate_epoch_cycle=None):
        """ A keras callback that prunes weights using a `LotteryTicketPruner`.
        Per the intention of lottery ticket pruning the model being trained is pruned at the beginning of every epoch so
        that training is done with the pruned weights set to zero.
        After completion of training the model will also be pruned so that the final trained model has pruning applied
        for inference.
        :param pruner: A `LotteryTicketPruner` instance that is used to prune weights during and just after training.
            The pruner is only used to apply the pruning mask to the model's weights. The caller should make sure that
            this pruner instance's `LotteryTicketPruner.prune_weights()` has been called to calculate the pruning mask.
        :param use_dwr: If True then the callback will apply Dynamic Weight Rescaling (DWR) to the unpruned weights in
            the model after every epoch.
            See section 5.2, "Dynamic Weight Rescaling" of https://arxiv.org/pdf/1905.01067.pdf.
            A quote from that paper describes it best:
                "For each training iteration and for each layer, we multiply the underlying weights by the ratio of the
                total number of weights in the layer over the number of ones in the corresponding mask."
        """
        super().__init__()
        self.pruner = pruner
        self.use_dwr = use_dwr
        self.prune_every_batch_iteration = prune_every_batch_iteration
        self.iterative_model_non_zero_dense_weights_after_pruning = iterative_model_non_zero_dense_weights_after_pruning
        self.iterative_model_non_zero_convolutional_weights_after_pruning = iterative_model_non_zero_convolutional_weights_after_pruning
        self.recalculate_epoch_cycle = recalculate_epoch_cycle
        if self.iterative_model_non_zero_dense_weights_after_pruning is not None:
            assert self.iterative_model_non_zero_convolutional_weights_after_pruning is not None
            assert isinstance(self.iterative_model_non_zero_dense_weights_after_pruning, List)
            assert isinstance(self.iterative_model_non_zero_convolutional_weights_after_pruning, List)
            assert isinstance(recalculate_epoch_cycle, int)
            assert len(self.iterative_model_non_zero_convolutional_weights_after_pruning) == len(self.iterative_model_non_zero_dense_weights_after_pruning)

        self.pruning_iteration = 0

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        # Prune weights after training is completed so inference uses pruned weights
        self.pruner.apply_pruning(self.model)
        # Don't apply DWR at the end of training since it changes the weights that we just trained so hard to arrive at

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        # End of epoch so prune the weights that we're pruning
        self.pruner.apply_pruning(self.model)
        if self.use_dwr:
            self.pruner.apply_dwr(self.model)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.iterative_model_epoch_cyle == 0 and epoch > 0:
            smallest_weight_pruner = ltp.LotteryTicketPruner(self.model)

            smallest_weight_pruner.calc_prune_mask(self.model, self.iterative_model_non_zero_dense_weights_after_pruning[self.pruning_iteration],
                                        self.iterative_model_non_zero_convolutional_weights_after_pruning[self.pruning_iteration],
                                        "smallest_weights_layer_dependent_pruning_percentage")

            smallest_weight_pruner.apply_pruning(self.model)
          
        self.pruning_iteration += 1

    def on_train_batch_end(self, batch, logs=None):
        if self.prune_every_batch_iteration:
            super().on_epoch_begin(batch, logs)
            self.pruner.apply_pruning(self.model)
            if self.use_dwr:
                self.pruner.apply_dwr(self.model)
