import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import argparse
from datetime import datetime
import mydataloader as dataloader
import torch.optim.lr_scheduler
import pandas as pd
from function import *
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=75, type=int)
parser.add_argument('--wdecay', default=5e-4, type=float, help='initial learning rate')

parser.add_argument('--noise_mode',  default='instance', choices=['sym', 'pair', 'instance'])
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--T', default=0.2, type=float, help='confidence threshold')
parser.add_argument('--k', default=3, type=int, help='queue length')

parser.add_argument('--seed_model', default=1, type=int)
parser.add_argument('--seed_noise', default=1, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--data_path', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--semi', default='no', type=str)
parser.add_argument('--model', default='resnet18', type=str)
parser.add_argument('--without_R', default=0, type=int) # for table 7 reproduction
parser.add_argument('--without_Lcr', default=0, type=int) # for table 7 reproduction
parser.add_argument('--save_sel_sam', default=0, type=int) # for Fig. 10 reproduction
parser.add_argument('--fig_7', default=1, type=int) # for Fig. 7 reproduction
parser.add_argument('--fluctuation_ablation', default=0, type=int)


args = parser.parse_args()

if args.dataset == 'cifar10':
    args.data_path = './dataset/cifar10'
    args.num_class = 10
elif args.dataset == 'cifar100':
    args.data_path = './dataset/cifar100'
    args.num_class = 100
print(args)

if args.fig_7:
    args.num_epochs=10
set_env(args)

def build_model():
    if args.model == 'resnet32':
        from model.resnet32 import resnet32
        model = resnet32(args.num_class)
        print('============ use resnet32 ')
    elif args.model == 'resnet18':
        from model.resnet import ResNet18
        model = ResNet18(args.num_class)
        print('============ use resnet18 ')
    elif args.model == 'resnet34':
        from model.resnet import ResNet34
        model = ResNet34(args.num_class)
        print('============ use resnet34 ')
    model = model.cuda()
    return model

def main():
    test_log = open('./log/base/%s_%s_%.1f_k=%d_seedmodel=%d' % (args.dataset, args.noise_mode, args.r, args.k, args.seed_model) + '_test_fig7.txt', 'w')
    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=8, root_dir=args.data_path, args=args,
                                         noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))
    net = build_model()
    memory_bank = []
    best_acc = 0.0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60], 0.1)

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    warmup_trainloader = loader.run('warmup')

    epoch_list = [] # for Fig. 10 reproduction
    num_sel_sam_list = [] # for Fig. 10 reproduction
    num_clean_count_list = [] # for Fig. 10 reproduction

    for epoch in range(args.num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        if epoch < args.warm_up:
            print('============ Warmup stage | lr = %.3f, T in penalty = %.3f' % (lr, args.T))
            _,  memory_bank = eval_train(net, memory_bank, eval_loader, args, epoch, test_log)
            warmup(epoch, net, optimizer, warmup_trainloader, args)
        else:
            print('============ Train stage | lr = %.3f, T in penalty = %.3f' % (lr, args.T))
            if args.save_sel_sam: # for Fig. 10 reproduction
                prob, pred, memory_bank, num_sel_sam = eval_train(net, memory_bank, eval_loader, args, epoch, test_log)
                labeled_trainloader, clean_count = loader.run('train', pred, prob, test_log)

                epoch_list.append(epoch) # sample selection before n-epoch training
                num_sel_sam_list.append(num_sel_sam)
                num_clean_count_list.append(clean_count)
            
            else: # default code
                prob, pred, memory_bank = eval_train(net, memory_bank, eval_loader, args, epoch, test_log)
                if args.fig_7:
                    labeled_trainloader = loader.run('labeled_fig7', pred, prob, test_log)
                    fig_7(net, labeled_trainloader, args)
                labeled_trainloader = loader.run('train', pred, prob, test_log)

            train(epoch, net, optimizer, labeled_trainloader, args)

        test_acc = test(epoch, net, test_loader, test_log)
        print('\n')
        if test_acc > best_acc:
            best_acc = test_acc
        sch_lr.step()
    print('best test Acc: ', best_acc)
    test_log.write('Best Accuracy:%.2f\n' % (best_acc))

    if args.save_sel_sam:
        if args.fluctuation_ablation:
            fig10_to_csv(epoch_list, num_sel_sam_list, num_clean_count_list, 
                        filename=f"./fluctuation_ablation/selection_history_dataset_{args.dataset}_noiserate_{args.r}_noisemode_{args.noise_mode}_seed_{args.seed_model}_R_{args.without_R}_Lcr_{args.without_Lcr}.csv") # for Fig. 10 reproduction
        else:
            fig10_to_csv(epoch_list, num_sel_sam_list, num_clean_count_list, 
                        filename=f"./figure_10/selection_history_dataset_{args.dataset}_noisemode_{args.noise_mode}_seed_{args.seed_model}_R_{args.without_R}_Lcr_{args.without_Lcr}.csv") # for Fig. 10 reproduction
     


if __name__ == '__main__':
    main()