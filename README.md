
# Reproducing  "Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization"

We utilize [the official PyTorch Implementation of Self-Filtering](https://github.com/1998v7/Self-Filtering) provided by authors.

> Paper ["Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization"](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900511.pdf) is accepted to **ECCV 2022**.

# Experiment setting for reproducing

## Scope of reproducibility

- **Claim 1 : SFT outperforms the almost state-of-the-art on CIFAR-10 with three noise types. (Table 1)**
- **Claim 2 : The regularization term in the warm-up stage effectively separates clean and noisy samples. (Fig 7)**
- **Claim 3 : The regularization terms improve the performance of SFT. (Table 7, Fig 10)**
- **Claim 4 : With a smaller T, SFT attains the best performance. (Fig 9)**

To verify whether the model can robustly learn despite changes in initial model parameters and data distribution,
we conducted additional experiments by changing only the data seed while keeping the model seed constant. Upon
reviewing the code provided in the paper, we found discrepancies between the provided code and the method described
in the paper. This is explained in the following 2.
1. To verify whether the model can robustly learn despite changes in initial model parameters and data distribution,
we conducted additional experiments by changing only the data seed while keeping the model seed constant.
2. The authors proposed fluctuation, where samples are considered noisy if they transition from correct predictions
to incorrect predictions. However, upon examining the implementation in the open-sourced code, we found
that selection is based on both the prediction probabilities stored in the memory bank and the presence of
fluctuation. Therefore, we implemented the fluctuation criterion as described in the paper and conducted
experiments

### Add new argument for reproduction

- `--without_R`, `--without_Lcr` for table 7, Fig. 10 reproduction

- `--save_sel_sam` for Fig.10 reproduction

- `--seed_noise` and `--seed_model` arguments were created to separate the model seed and dataset configuration seed, which were combined as a single seed in the original code. For all experiments except Ablation 1, `--seed_noise` is kept constant while `--seed_model` is varied for repeated experiments. Only in Ablation 1 is `--seed_model` kept constant, and `--seed_noise` varied for repeated testing.

- `--fluctuation_ablation` for our implementation for fluctuation algorithm based on paper.

- we utilize [Coteaching official code](https://github.com/bhanML/Co-teaching) and [JoCoR official code](https://github.com/hongxin001/JoCoR) for reproduction Claim 1.


### Hyper-parameter and settings

`k`  denotes memory bank size. It can be set as `[2,3,4]`. For all experiment, we set it as `3` follow original setting.

`T`  denotes threshold in confidence penalty. For all experiment, we set it as `0.2`

For CIFAR-10, `warm_up = 10`,`model = resnet18`, `num_epochs = 75`

For CIFAR-100, `warm_up = 30`,`model = resnet34`, `num_epochs = 100`

**In our study, we focused on CIFAR-10 for simplification of the experiment.**
To facilitate experimentation, please download the CIFAR-10 dataset in `./datset` folder before running.

### Run SFT

```
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 75 --noise_mode instance --r 0.2 --k 2 --T 0.2 --gpuid 0
```

### Run for Claim 1 : Test accuracy on CIFAR-10 (Table 1. Result)

If you want to reproduct Claim 1, run below.

```
./Co-teaching-master/table1_coteaching.sh
```
```
./JoCOR-master/table1_jocor.sh
```
```
./sh_files/table1.sh
```

Running `table1_coteaching.sh` in the Co-teaching-master directory, `table1_jocor.sh` in the JoCoR-master directory, and `table1.sh` in the sh_files directory will store the performance values of each model in the log folder in the form of txt files. Each txt file contains the test accuracy values and the best accuracy values for each epoch under different conditions. To verify the results of `Table 1` in authors' paper, you can calculate the average and standard deviation using the best accuracy values stored in the `.txt` files.


### Run for Claim 2 : Effect of regularization term in Warm-up stage (Fig.7 Result)
If you want to reproduct Claim 2, run below.

```
## to get loss with warm up regularizer
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 75 --noise_mode instance --r 0.4 --k 2 --T 0.2 --gpuid 0 --without_R 0

## to get loss without warm up regularizer
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 75 --noise_mode instance --r 0.4 --k 2 --T 0.2 --gpuid 0 --without_R 1
```

To check Figure 7, I have added code to the fig_7 function in the `function.py` file to save the clean loss and noise loss both with and without the warm-up regularizer.\
To save the results with and without regularization, you need to set the `--without_R` argument.\
If you want to remove the regularizer term during the warm-up process, set `--without_R` to 1.\
The results are saved as pickle files in the `./figure_code` folder.\
To verify the results, run all the cells in the `fig7_plot.ipynb` file located in the figure_code folder.


### Run for Claim 3 : Ablation study of each regularization term (Table 7, Fig. 10 Result)
If you want to reproduct Claim 3, run below.

To facilitate experimentation, please create the `./figure_10` folder before running. This folder contains performance records (`./figure_10/log`) and selection history.

```
./sh_files/figure_10_reproduct.sh

```
After running this Shell Script file, you can view `Figure 2` from our paper in the following `./figure_code/figure_10.ipynb` file.


### Run for Claim 4 : Study of Memory bank Size (Fig. 9)
If you want to reproduct Claim 4, run below.


```
./sh_files/figure_9_reproduct.sh
```

Or if you want to run with specific bank size,
```
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 75 --noise_mode instance --r 0.2 --k {bank_size_you_want} --T 0.2 --gpuid 0
```
---
### Run for ablation 1 : Robustness in various data distribtion
Specify the number you want for the seed_noise argument when running the script.

(Our experiments were conducted with seed numbers 1, 2, 3, 4, and 5.)
```
python main.py --dataset cifar10 --model resnet18 --batch_size 32 --lr 0.02 --warm_up 10 --num_epochs 75 --noise_mode instance --r 0.2 --k 2 --T 0.2 --gpuid 0 --seed_noise {random_seed_noise_you_want}
```

### Run for ablation 2 : Study about our fluctuation implementation based on paper
If you want to reproduct ablation 2, run below.

To facilitate experimentation, please create the `./fluctuation_ablation` folder before running. This folder contains performance records (`./figure_10/log`) and selection history.
 
```
./sh_fil/fluctu
```ation_ablation.sh
```
After running this Shell Scpt file, you can view `Figure 4` from the paper in the following `./figure_code/fluctuation_figure.ipynb` file.

---
### Cite
```
@inproceedings{wei2022self,
  title={Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization},
  author={Wei, Qi and Sun, Haoliang and Lu, Xiankai and Yin, Yilong},
  booktitle={European Conference on Computer Vision},
  pages={516--532},
  year={2022},
  organization={Springer}
}
```
