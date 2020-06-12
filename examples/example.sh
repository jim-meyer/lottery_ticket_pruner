#!/bin/bash

python example.py --iterations 5 --epochs 100 --which_set mnist --prune_strategy smallest_weights
python example.py --iterations 5 --epochs 100 --which_set mnist --prune_strategy smallest_weights_global
python example.py --iterations 5 --epochs 100 --which_set mnist --prune_strategy large_final

python example.py --iterations 5 --epochs 100 --which_set cifar10 --prune_strategy smallest_weights
python example.py --iterations 5 --epochs 100 --which_set cifar10 --prune_strategy smallest_weights_global
python example.py --iterations 5 --epochs 100 --which_set cifar10 --prune_strategy large_final

python example.py --iterations 5 --epochs 100 --which_set cifar10_reduced_10x --prune_strategy smallest_weights
python example.py --iterations 5 --epochs 100 --which_set cifar10_reduced_10x --prune_strategy smallest_weights_global
python example.py --iterations 5 --epochs 100 --which_set cifar10_reduced_10x --prune_strategy large_final

# Again using Dynamic Weight Rescaling
python example.py --iterations 5 --epochs 100 --dwr --which_set mnist --prune_strategy smallest_weights
python example.py --iterations 5 --epochs 100 --dwr --which_set mnist --prune_strategy smallest_weights_global
python example.py --iterations 5 --epochs 100 --dwr --which_set mnist --prune_strategy large_final

python example.py --iterations 5 --epochs 100 --dwr --which_set cifar10 --prune_strategy smallest_weights
python example.py --iterations 5 --epochs 100 --dwr --which_set cifar10 --prune_strategy smallest_weights_global
python example.py --iterations 5 --epochs 100 --dwr --which_set cifar10 --prune_strategy large_final

python example.py --iterations 5 --epochs 100 --dwr --which_set cifar10_reduced_10x --prune_strategy smallest_weights
python example.py --iterations 5 --epochs 100 --dwr --which_set cifar10_reduced_10x --prune_strategy smallest_weights_global
python example.py --iterations 5 --epochs 100 --dwr --which_set cifar10_reduced_10x --prune_strategy large_final
