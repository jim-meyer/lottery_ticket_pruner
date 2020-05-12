# Lottery Ticket Pruner

Deep Neural Networks (DNNs) can often benefit from "pruning" some weights in the network, turning dense matrices of weights
into sparse matrices of weights with little or no loss in accuracy of the overall model. *In fact, these pruned networks
often times result in better model accuracy!*

# Installation

    pip install lottery_ticket_pruner

# Usage

# Building

See https://packaging.python.org/tutorials/packaging-projects/

    pip install -r requirements.txt
    python setup.py sdist bdist_wheel
    TODO - etc etc etc

# Testing

Running unit tests is done via `tox` and automatically generates a code coverage report.

    tox

# Contributing

Pull requests to [this repo](https://github.com/jim-meyer/lottery_ticket_prune) are always welcome.

# Citations

Inspired from these papers:

[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)

[Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)
