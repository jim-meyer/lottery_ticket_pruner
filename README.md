# Lottery Ticket Pruner

Deep Neural Networks (DNNs) can often benefit from "pruning" some weights in the network, turning dense matrices of weights
into sparse matrices of weights with little or no loss in accuracy of the overall model. *In fact, these pruned networks
often times result in better model accuracy!*

This is a keras implementation of the pruning strategies outlined in two papers:

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)
- [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)


# Installation

    cd <root directory of this repo>
    pip install -e .

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

For a full working example that computes the accuracy for an *untrained* model that has been pruned, as well
as training a model from scratch using lotttery ticket pruning, see the [example code](example/example.py).
This example code uses the [MNIST](https://keras.io/api/datasets/mnist/) and
[CIFAR10](https://keras.io/api/datasets/cifar10/) datasets.

# Building

See https://packaging.python.org/tutorials/packaging-projects/

    pip install -r requirements.txt
    python setup.py sdist bdist_wheel
    TODO - etc etc etc

# Testing

Running unit tests is done via [tox](https://pypi.org/project/tox/). This automatically generates a code coverage report too.

    tox

# Contributing

Pull requests to [this repo](https://github.com/jim-meyer/lottery_ticket_prune) are always welcome.
