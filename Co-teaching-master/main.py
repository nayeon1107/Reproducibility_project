# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from model import CNN
import argparse, sys
import numpy as np
from tqdm import tqdm

import datetime
import shutil

from loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '/home/user/Desktop/jeongwon/딥러닝 팀플/log/co-teaching')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[ symmetric, pairflip, instance ]', default='instance')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'cifar10, cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=75)
parser.add_argument('--seed_model', type=int, default=1)
parser.add_argument('--seed_noise', type=int, default=1)
parser.add_argument('--printfreq', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpuid', type=int, default=1)


args = parser.parse_args()

# Seed
torch.manual_seed(args.seed_model)
torch.cuda.manual_seed(args.seed_model)
torch.cuda.set_device(args.gpuid)
# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr 

# load dataset

if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    train_dataset = CIFAR10(root='/home/user/Desktop/jeongwon/딥러닝 팀플/dataset/cifar10',
                                download=False,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                random_state=args.seed_noise
                           )
    
    test_dataset = CIFAR10(root='/home/user/Desktop/jeongwon/딥러닝 팀플/dataset/cifar10',
                                download=False,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                random_state=args.seed_noise
                          )

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 100
    train_dataset = CIFAR100(root='/home/user/Desktop/jeongwon/딥러닝 팀플/dataset2/cifar100',
                            download=False,  
                            train=True, 
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate,
                            random_state=args.seed_noise
                            )
    
    test_dataset = CIFAR100(root='/home/user/Desktop/jeongwon/딥러닝 팀플/dataset2/cifar100',
                                download=False,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                random_state=args.seed_noise
                            )

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir

model_str=args.dataset+'_'+args.noise_type+'_'+str(args.noise_rate)+'_seedmodel='+str(args.seed_model)

txtfile=save_dir+"/"+model_str+".txt"

def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    _, predicted = torch.max(output, 1)

    correct = 0
    total = 0
    total += target.size(0)
    correct += predicted.eq(target).cpu().sum().item()
    
    accu = 100. * correct / total
    return accu

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
    # print('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        # if i>args.num_iter_per_epoch:
        #     break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        accu1 = accuracy(logits1, labels)
        train_total+=1
        train_correct+=accu1

        logits2 = model2(images)
        accu2 = accuracy(logits2, labels)
        train_total2+=1
        train_correct2+=accu2
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.printfreq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, accu1, accu2, loss_1.data.item(), loss_2.data.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    # print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn1.cuda()
    # print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn2.cuda()
    # print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0

    with open(txtfile, "w") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')
    
    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    # print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results

    best_acc = 0.0
    # training
    for epoch in tqdm(range(1, args.n_epoch)):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)

        mean_accu = (test_acc1 + test_acc2)/2
        if mean_accu > best_acc:
            best_acc = mean_accu
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        # print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write('Mean Accuracy For : ' + str(int(epoch)) + ': '  + str(mean_accu) + "\n")

    with open(txtfile, "a") as myfile:
        myfile.write('best Accuracy is : ' + str(best_acc) + "\n")

if __name__=='__main__':
    main()
