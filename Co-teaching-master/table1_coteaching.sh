#!/bin/bash

for dataset in cifar10; do
    for noise_type in instance; do
        for seed in 1 2 3 4 5; do
            for noise_rate in 0.2 0.4; do
                    # Run Python script with specific parameters and append results to the log file
                    python main.py --dataset $dataset --noise_type $noise_type --noise_rate $noise_rate --seed_model $seed
            done
        done
    done
done