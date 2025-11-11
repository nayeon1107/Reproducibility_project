import sys, torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import random
from loss import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#%%

def set_env(args):
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed_model)
    torch.manual_seed(args.seed_model)
    torch.cuda.manual_seed_all(args.seed_model)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def one_lossF(output, one_hot, fig_7):
    log_prob = torch.nn.functional.log_softmax(output, dim=1)
    if fig_7 == False:
        loss = - torch.sum(log_prob * one_hot) / output.size(0)
    else:
        loss = - torch.sum(log_prob * one_hot, dim=1)
    return loss


class penalty_f2(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, outputs, targets, args, mode='warm_up'):
        loss_ce = F.cross_entropy(outputs, targets)
        bs = outputs.size()[0]
        pseudo_label = torch.softmax(outputs, dim=1)
        if mode == 'warm_up':
            if not self.args.without_Lcr:
                values, index = pseudo_label.topk(k=2, dim=1)
                latent_label = index[:, 1]
                latent_lam = torch.zeros(pseudo_label.size(0), 1).cuda().float()

                for i in range(values.size(0)):
                    x = values[i]
                    latent_lam[i] = max(args.T - min(x) / max(x), 0.0)
                    # 0.5 ????????
                conf_penalty = 0.5 * (F.cross_entropy(outputs, latent_label, reduction='none') * latent_lam).mean()
                loss = loss_ce + conf_penalty
            else:
                loss = loss_ce
        elif mode == 'train':
            if not self.args.without_Lcr:
                latent_labels = torch.zeros((bs, args.num_class)).cuda()
                for i in range(bs):
                    confident = pseudo_label[i]
                    max_ = confident[targets[i]]
                    confident = args.T - confident / max_
                    mask = (confident >= 0.0)
                    latent_labels[i] = confident * mask
                    # 0.1 은 람다 같은데, 람다를 논문에서 말한거처럼 1로 설정하면 성능은? table1의 SFT 결과랑 비슷하게 나와야함
                conf_penalty = 0.1 * one_lossF(outputs, latent_labels, fig_7=False) / (args.num_class - 1)    
                loss = loss_ce + conf_penalty            
            else:
                loss = loss_ce
        
        else:
            raise ValueError('')

        return loss

class penalty_f2_fig7(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, outputs, targets, args, mode='warm_up'):
        loss_ce = F.cross_entropy(outputs, targets, reduction='none')
        bs = outputs.size()[0]
        pseudo_label = torch.softmax(outputs, dim=1)

        if mode == 'warm_up':
            if not self.args.without_Lcr:
                values, index = pseudo_label.topk(k=2, dim=1)
                latent_label = index[:, 1]
                latent_lam = torch.zeros(pseudo_label.size(0), 1).cuda().float()

                for i in range(values.size(0)):
                    x = values[i]
                    latent_lam[i] = max(args.T - min(x) / max(x), 0.0)
                    
                conf_penalty = 0.5 * (F.cross_entropy(outputs, latent_label, reduction='none') * latent_lam)
                loss = loss_ce + conf_penalty
            else:
                loss = loss_ce
        elif mode == 'train':
            if not self.args.without_Lcr:
                latent_labels = torch.zeros((bs, args.num_class)).cuda()
                for i in range(bs):
                    confident = pseudo_label[i]
                    max_ = confident[targets[i]]
                    confident = args.T - confident / max_
                    mask = (confident >= 0.0)
                    latent_labels[i] = confident * mask
                    # 0.1 은 람다 같은데, 람다를 논문에서 말한거처럼 1로 설정하면 성능은? table1의 SFT 결과랑 비슷하게 나와야함
                conf_penalty = 0.1 * one_lossF(outputs, latent_labels, fig_7=False) / (args.num_class - 1)    
                loss = loss_ce + conf_penalty            
            else:
                loss = loss_ce
        else:
            raise ValueError('')

        return loss
def warmup(epoch, net, optimizer, dataloader, args):
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = penalty_f2(args)

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs= net(inputs)
        if args.without_R : # without R
            L = CEloss(outputs, labels)
        
        elif not args.without_R:
            if args.noise_mode == 'instance' or args.noise_mode == 'pair':  # penalize confident prediction for pair and inst noise
                L = conf_penalty(outputs, labels, args, mode='warm_up')
            else:
                L = CEloss(outputs, labels)

        L.backward()
        optimizer.step()
        if batch_idx % 50 == 0 :
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f' %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, L.item()))
            sys.stdout.flush()


def eval_train(model, memory_bank, eval_loader, args, epoch, test_log):
    if epoch >= args.warm_up:
        selection = True
    else:
        selection = False

    model.eval()
    correct = 0
    total = 0

    if selection:
        memory_bank_last = memory_bank[-1]
        fluctuation_ = [0] * 50000

    result = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            max_probs, pred = outputs.max(1)

            for b in range(inputs.size(0)):
                sample_index = index[b]
                if args.fluctuation_ablation:
                    '''
                    result : a list that stores whether predictions are correct or not.
                    '''
                    if pred[b] == targets[b]:
                        result[sample_index] = 1
                    else:
                        result[sample_index] = 0

                    if selection:
                        if memory_bank_last[sample_index] == 1 and result[sample_index] == 0:
                            fluctuation_[sample_index] = 0 # We defined occuring fluctuation, set 0
                        else:
                            fluctuation_[sample_index] = 1

                else: #defualt code
                    if pred[b] == targets[b]:
                        result[sample_index] = max_probs[b]
                    else:
                        result[sample_index] = 0
                    if selection:
                        # calculate the fluctuation
                        if memory_bank_last[sample_index] > result[sample_index] and result[sample_index] == 0:
                            # the prediction of this sample changes from correct to wrong, thus it is fluctuation. we consider it as the clean with 0%
                            fluctuation_[sample_index] = 0
                        else:
                            fluctuation_[sample_index] = 1
            total += targets.size(0)
            correct += pred.eq(targets).cpu().sum().item()
    print('Epoch %d | Accuracy on train set: %.2f%% ' % (epoch, 100. * correct / total))
    test_log.write('Epoch %d | Accuracy on train set: %.2f%% ' % (epoch, 100. * correct / total))

    # In practice, the fluctuation of predictions is easily influenced by SGD optimizer especially in extreme noise ratio,
    # Here, we design a smooth way by adding the confidence of prediction to fluctuation.
    # For that, the criterion will select the sample with high confidence even if there is a fluctuation
    if selection:
        for i in range(args.k - 1):
            memory_bank[i] = memory_bank[i + 1]
        memory_bank[-1] = result

        if args.fluctuation_ablation:
            prob = np.array(fluctuation_)
            pred = (prob > 0.5)  # fluctuation is 0, then unselect

        else: #default code
            confidence_smooth = torch.zeros_like(memory_bank[0])
            for i in range(args.k):
                confidence_smooth += memory_bank[i]
            prob = (confidence_smooth.numpy() + np.array(fluctuation_)) / (args.k + 1)  # adding confidence make fluctuation more smooth
            pred = (prob > 0.5) 
        
        if args.save_sel_sam:
            pred_idx = pred.nonzero()[0]
            num_sel_sam = len(pred_idx)
            return prob, pred, memory_bank, num_sel_sam

        return prob, pred, memory_bank

    else:        
        if len(memory_bank) < args.k:
            memory_bank.append(result)
        else:
            for i in range(args.k - 1):
                memory_bank[i] = memory_bank[i + 1]
            memory_bank[-1] = result
        return None, memory_bank


def test(epoch, net1, test_loader, test_log):
    net1.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    print("\n| Test Epoch %d\t Accuracy: %.2f%% \n" % (epoch, 100. * correct / total))
    test_log.write("\n| Test Epoch %d\t Accuracy: %.2f%% \n" % (epoch, 100. * correct / total))
    return 100. * correct / total


def train(epoch, net, optimizer, labeled_trainloader, args):
    net.train()
    conf_penalty = penalty_f2(args)

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    correct = 0
    total = 0
    for batch_idx, (inputs_x, _, label_x, _) in enumerate(labeled_trainloader):
        inputs_x, label_x = inputs_x.cuda(), label_x.cuda()

        outputs_x = net(inputs_x)
        loss = conf_penalty(outputs_x, label_x, args, mode='train')
        _, predicted = torch.max(outputs_x, 1)

        total += label_x.size(0)
        correct += predicted.eq(label_x).cpu().sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t  loss: %.2f'% (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
            sys.stdout.flush()

def fig_7(net, labeled_trainloader, args):
    net.eval()
    conf_penalty = penalty_f2_fig7(args)
    noise_loss_list, clean_loss_list = [], []
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs_x, _, label_x, _, noise) in enumerate(labeled_trainloader):
            inputs_x, label_x = inputs_x.cuda(), label_x.cuda()

            outputs_x = net(inputs_x)
            loss = conf_penalty(outputs_x, label_x, args, mode='train')

            noise_loss_list.extend(loss[noise==0].cpu().tolist())
            clean_loss_list.extend(loss[noise==1].cpu().tolist())

        # figure 그리는 directory에 각 loss들의 pickle 파일 저장
        if args.without_R:
            with open('./figure_code/noise_loss_list_withoutR.pickle', 'wb') as f:
                pickle.dump(noise_loss_list, f, pickle.HIGHEST_PROTOCOL)
            with open('./figure_code/clean_loss_list_withoutR.pickle', 'wb') as f:
                pickle.dump(clean_loss_list, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('./figure_code/noise_loss_list_withR.pickle', 'wb') as f:
                pickle.dump(noise_loss_list, f, pickle.HIGHEST_PROTOCOL)
            with open('./figure_code/clean_loss_list_withR.pickle', 'wb') as f:
                pickle.dump(clean_loss_list, f, pickle.HIGHEST_PROTOCOL)

def fig10_to_csv(epoch_list, num_sel_sam_list, num_clean_count_list, filename="output.csv"): 
    # for Fig. 10 reproduction
    data = {
        'epoch': epoch_list,
        'num_total_selection': num_sel_sam_list,
        'num_clean': num_clean_count_list
    }
    df = pd.DataFrame(data)

    # CSV 파일로 저장
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")