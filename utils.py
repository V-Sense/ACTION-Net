import csv
import os
import torch
import pdb
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_checkpoint(save_dir, filename):
    checkpoint = torch.load(os.path.join(save_dir, filename), map_location='cpu')
    return checkpoint

def save_checkpoint(net, optimizer, filename):
    checkpoint = {
            'state_dict': net.module.state_dict(),
            'optimizer' : optimizer.state_dict()
            }
    torch.save(checkpoint, filename)


# def save_checkpoint(net, filename):
#     checkpoint = {
#             'state_dict': net.module.state_dict()
#             }
#     torch.save(checkpoint, filename)


# def save_checkpoint(net, optimizer, scheduler, filename):
#     checkpoint = {
#             'state_dict': net.module.state_dict(),
#             'optimizer' : optimizer.state_dict(),
#             'scheduler': scheduler.state_dict()}
#     torch.save(checkpoint, filename)



def adjust_learning_rate(learning_rate, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # import pdb; pdb.set_trace()

    gamma = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    base_lr = learning_rate * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * param_group['lr_mult']





# def adjust_learning_rate(learning_rate, optimizer, epoch, lr_steps):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr_new = learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr_new


# def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     # import pdb; pdb.set_trace()
#     if lr_type == 'step':
#         gamma = 0.1 ** (sum(epoch >= np.array(lr_steps)))
#         base_lr = args.lr * gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = base_lr * param_group['lr_mult']
#     elif lr_type == 'cos':
#         import math
#         gamma = 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
#         base_lr = args.lr * gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = base_lr * param_group['lr_mult']
#     else:
#         raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy_ForIcarl(outputs, targets):
    batch_size = [*outputs.shape][0]
    correct = outputs.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum()
    return (n_correct_elems / batch_size).item()

def calculate_accuracy(outputs, targets):
    batch_size = [*outputs.shape][0]
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum()
    return (n_correct_elems / batch_size).item()








# def calculate_accuracy_topk(outputs, targets, topk):
#     batch_size = targets.size(0)
#     _, pred = outputs.topk(topk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(targets.view(1, -1).expand_as(pred))
#     correct_k = correct[:topk].float().sum().item()
#     ret = correct_k / batch_size
#     return ret


def calculate_accuracy_topk(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        ret.append(correct_k / batch_size)

    return ret


def calculate_precision(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return precision_score(targets.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), average = 'macro')
