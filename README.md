# Lottery Ticket Pruner

Deep Neural Networks (DNNs) can often benefit from "pruning" some weights in the network, turning dense matrices of weights
into sparse matrices of weights with little or no loss in accuracy of the overall model.

This is a keras implementation of the most relevant pruning strategies outlined in two papers:

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)
- [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)

The pruning strategies implemented in this package can reduce the number of non-zero weights of CNNs, DNNs
by 40-98% with negligible losses in accuracy of the final model.
Various techniques like [MorphNet](https://ai.googleblog.com/2019/04/morphnet-towards-faster-and-smaller.html) can then
be applied to further optimize these now sparse models to decrease model size and/or inference times.

# Installation

    pip install lottery-ticket-pruner

# Usage

A typical use of the code in this repo looks something like this:

    from lottery_ticket_pruner import LotteryTicketPruner, PrunerCallback
    
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

For a full working example that computes the accuracy for an untrained model that has been pruned, as well
as training a model from scratch using lotttery ticket pruning, see the [example code](https://github.com/jim-meyer/lottery_ticket_pruner/examples/example.py).
This example code uses the [MNIST](https://keras.io/api/datasets/mnist/) and
[CIFAR10](https://keras.io/api/datasets/cifar10/) datasets.

# Results

To see the effects of pruning at 20%, 44.72%, 81.18%, 98.78%, 99.98% using the 3 supported pruning strategies across the
MNIST and CIFAR10 datasets, with and without Dynamic Weight Resizing (DWR) were obtained via:

To see the effects of pruning at 20%, 55.8%, 89.6%, 99.3% using the 3 supported pruning strategies across the
MNIST and CIFAR10 datasets, with and without Dynamic Weight Resizing (DWR) were obtained via:

    python examples/example.py --iterations 5 --epochs 100 --which_set 'mnist' --prune_strategy smallest_weights
    python examples/example.py --iterations 5 --epochs 100 --which_set 'mnist' --prune_strategy smallest_weights_global

    python examples/example.py --iterations 5 --epochs 100 --which_set 'mnist' --prune_strategy smallest_weights --dwr
    python examples/example.py --iterations 5 --epochs 100 --which_set 'mnist' --prune_strategy smallest_weights_global --dwr

    python examples/example.py --iterations 5 --epochs 100 --which_set 'cifar10' --prune_strategy smallest_weights
    python examples/example.py --iterations 5 --epochs 100 --which_set 'cifar10' --prune_strategy smallest_weights_global

    python examples/example.py --iterations 5 --epochs 100 --which_set 'cifar10' --prune_strategy smallest_weights --dwr
    python examples/example.py --iterations 5 --epochs 100 --which_set 'cifar10' --prune_strategy smallest_weights_global --dwr

The results of averaging across 5 iterations, removing the min and max results were as follows:

TODO - add results here

    |Prune Percentage|  |Dataset|   |Prune Strategy|            |DWR?|      |Avg Accuracy|  |Avg Epochs|
    |:---|              |:---|      |:---:|                     |:---:|     |:---:|         |:---:|
    |20%|               |mnist|     |smallest_weights|          |False|     ||              ||
    |20%|               |mnist|     |smallest_weights_global|   |False|     ||              ||
    |20%|               |cifar10|   |smallest_weights|          |False|     ||              ||
    |20%|               |cifar10|   |smallest_weights_global|   |False|     ||              ||

Using Dynamic Weight Reduction:

    |Prune Percentage|  |Dataset|   |Prune Strategy|            |DWR?|      |Avg Accuracy|  |Avg Epochs|
    |:---|              |:---|      |:---:|                     |:---:|     |:---:|         |:---:|
    |20%|               |mnist|     |smallest_weights|          |True|      ||              || 
    |20%|               |mnist|     |smallest_weights_global|   |True|      ||              ||
    |20%|               |cifar10|   |smallest_weights|          |True|      ||              || 
    |20%|               |cifar10|   |smallest_weights_global|   |True|      ||              ||


# Working In This Repo

The information in this section is only needed if you need to modify this package.

This repo uses Github Actions to perform [Continuous Integration checks, tests for each push, pull request](https://github.com/jim-meyer/lottery_ticket_pruner/actions).

Likewise, when a new release is tagged a new version of the package is automatically built and uploaded to [pypi](https://pypi.org).

## Local Testing

Running unit tests locally is done via [tox](https://pypi.org/project/tox/). This automatically generates a code coverage report too.

    tox

# FAQ

Q: The two papers cited above refer to more pruning strategies than are implemented here. When will you support the
XXX pruning strategy?

A: The goal of this repo is to provide an implementation of the more effective strategies described
by the two papers. If other effective strategies are developed then pull requests implementing those strategies are welcomed.

Q: Why isn't python 3.5 supported?

A: keras>=2.1.0, pandas>=1.0 don't support python 3.5. Hence this package does not either.

# Contributing

Pull requests to [this repo](https://github.com/jim-meyer/lottery_ticket_pruner) are always welcome.
