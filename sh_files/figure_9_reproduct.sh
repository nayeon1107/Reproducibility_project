for noisemode in instance; do
    for r in 0.2 0.4; do
        for banksize in 2 3 4 5; do
            log_file="./log/base/cifar10_${noise_mode}_${r}_k=${banksize}_test.txt"

            python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 100 --noise_mode $noisemode --r $r --k $banksize --T 0.2 --gpuid 0 | tee -a $log_file
        done
    done
done