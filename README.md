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

[examples/example.sh](https://github.com/jim-meyer/lottery_ticket_pruner/examples/example.py) was run to see the effects
of pruning at 20%, 55.78%, 89.6%, 99.3% using the 3 supported pruning strategies across the MNIST and CIFAR10 datasets.
Training was capped at 100 epochs to help control AWS expenses.

The results averaged across 3 iterations:

## MNIST (100 epochs)

    |Prune Percentage|  |Dataset|   |Prune Strategy|            |Avg Accuracy|
    |:---|              |:---|      |:---:|                     |:---:|
    |n/a|               |mnist|     |n/a|                       |0.937|
    |20%|               |mnist|     |large_final|               |0.935|
    |20%|               |mnist|     |smallest_weights|          |0.936|
    |20%|               |mnist|     |smallest_weights_global|   |0.939|
    |55.78%|            |mnist|     |large_final|               |0.936|
    |55.78%|            |mnist|     |smallest_weights|          |0.936|
    |55.78%|            |mnist|     |smallest_weights_global|   |0.939|
    |89.6%|             |mnist|     |large_final|               |0.936|
    |89.6%|             |mnist|     |smallest_weights|          |0.937|
    |89.6%|             |mnist|     |smallest_weights_global|   |0.939|
    |99.33%|            |mnist|     |large_final|               |0.936|
    |99.33%|            |mnist|     |smallest_weights|          |0.937|
    |99.33%|            |mnist|     |smallest_weights_global|   |0.939|

## CIFAR (100 epochs)

    |Prune Percentage|  |Dataset|   |Prune Strategy|            |Avg Accuracy|
    |:---|              |:---|      |:---:|                     |:---:|
    |n/a|               |cifar10|   |n/a|                       |0.427|
    |20%|               |cifar10|   |large_final|               |0.298|
    |20%|               |cifar10|   |smallest_weights|          |0.427|
    |20%|               |cifar10|   |smallest_weights_global|   |0.423|
    |55.78%|            |cifar10|   |large_final|               |0.294|
    |55.78%|            |cifar10|   |smallest_weights|          |0.427|
    |55.78%|            |cifar10|   |smallest_weights_global|   |0.424|
    |89.6%|             |cifar10|   |large_final|               |0.289|
    |89.6%|             |cifar10|   |smallest_weights|          |0.427|
    |89.6%|             |cifar10|   |smallest_weights_global|   |0.424|
    |99.33%|            |cifar10|   |large_final|               |0.288|
    |99.33%|            |cifar10|   |smallest_weights|          |0.428|
    |99.33%|            |cifar10|   |smallest_weights_global|   |0.425|

## CIFAR (500 epochs)

    |Prune Percentage|  |Dataset|   |Prune Strategy|            |Avg Accuracy|
    |:---|              |:---|      |:---:|                     |:---:|
    |n/a|               |cifar10|   |n/a|                       |0.550|
    |20%|               |cifar10|   |smallest_weights_global|   |0.550|
    |55.78%|            |cifar10|   |smallest_weights_global|   |0.552|
    |89.6%|             |cifar10|   |smallest_weights_global|   |0.554|
    |99.33%|            |cifar10|   |smallest_weights_global|   |0.554|

# Pruning the initial model weights with no training

One of the surprising findings of these papers is that if we simply *do inference on the model using the original weights,
with no training, but applying pruning the resulting models perform far (far!) better than a random guess*. Here are the
results of inference done after applying pruning to the random initial weights of the model without any training. The
initial model, used as an input to the pruning logic, was trained for 100 epochs.

## MNIST

    |Prune Percentage|  |Dataset|   |Prune Strategy|                        |Avg Accuracy|
    |:---|              |:---|      |:---:|                                 |:---:|
    |n/a|               |mnist|     |no pruning done - random weights|      |0.121|
    |n/a|               |mnist|     |source model trained for 100 epochs|   |0.936|
    |20%|               |mnist|     |large_final|                           |0.760|
    |20%|               |mnist|     |smallest_weights|                      |0.737|
    |20%|               |mnist|     |smallest_weights_global|               |0.722|
    |55.78%|            |mnist|     |large_final|                           |0.911|
    |55.78%|            |mnist|     |smallest_weights|                      |0.899|
    |55.78%|            |mnist|     |smallest_weights_global|               |0.920|
    |89.6%|             |mnist|     |large_final|                           |0.744|
    |89.6%|             |mnist|     |smallest_weights|                      |0.703|
    |89.6%|             |mnist|     |smallest_weights_global|               |0.925|
    |99.33%|            |mnist|     |large_final|                           |0.176|
    |99.33%|            |mnist|     |smallest_weights|                      |0.164|
    |99.33%|            |mnist|     |smallest_weights_global|               |0.098|

## CIFAR

    |Prune Percentage|  |Dataset|   |Prune Strategy|                        |Avg Accuracy|
    |:---|              |:---|      |:---:|                                 |:---:|
    |n/a|               |cifar10|   |no pruning done - random weights|      |0.094|
    |n/a|               |mnist|     |source model trained for 100 epochs|   |0.424|
    |20%|               |cifar10|   |large_final|                           |0.232|
    |20%|               |cifar10|   |smallest_weights|                      |0.180|
    |20%|               |cifar10|   |smallest_weights_global|               |0.201|
    |55.78%|            |cifar10|   |large_final|                           |0.192|
    |55.78%|            |cifar10|   |smallest_weights|                      |0.240|
    |55.78%|            |cifar10|   |smallest_weights_global|               |0.251|
    |89.6%|             |cifar10|   |large_final|                           |0.101|
    |89.6%|             |cifar10|   |smallest_weights|                      |0.102|
    |89.6%|             |cifar10|   |smallest_weights_global|               |0.240|
    |99.33%|            |cifar10|   |large_final|                           |0.100|
    |99.33%|            |cifar10|   |smallest_weights|                      |0.099|
    |99.33%|            |cifar10|   |smallest_weights_global|               |0.100|

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
