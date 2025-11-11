#!/bin/bash

for dataset in cifar10; do
    for noise_mode in instance; do
        for seed in 1 2 3 4 5; do
            for noise_rate in 0.2 0.4; do
                    # Define log file path
                    log_file="../log/base/${dataset}_${noise_mode}_${noise_rate}_seedmodel=${seed}_test.txt"
                    # Run Python script with specific parameters and append results to the log file
                    python ../main.py --dataset $dataset --model resnet18 --batch_size 32 --warm_up 10 --num_epochs 75 --learning_rate 0.02 --noise_mode $noise_mode --r $noise_rate --seed_model $seed --gpuid 0 --without_R 0 --without_Lcr 0 | tee -a $log_file
                done
            done
        done
    done
done