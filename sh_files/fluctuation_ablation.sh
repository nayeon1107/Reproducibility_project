#!/bin/bash


for noise_rate in 0.4; do
    for noise_mode in sym; do
        for seed in 1 2 3 4 5; do
            for fluctuation_ablation in 1; do
                log_file="../fluctuation_ablation/log/cifar10_noiserate_${noise_rate}_noise_${noise_mode}_seed_${seed}.txt"
                
                # Run Python script with specific parameters and append results to the log file
                python ../main.py --dataset cifar10 --model resnet18 --fluctuation_ablation $fluctuation_ablation --save_sel_sam 1 --batch_size 32 --warm_up 10 --num_epochs 75 --learning_rate 0.02 --noise_mode $noise_mode --r $noise_rate --seed_model $seed --gpuid 0 --fig_7 0 | tee -a $log_file
            done    
        done
    done
done