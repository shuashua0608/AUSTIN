import os
import torch
import re


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (16 * 1) -> (1 * 16)

    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)
    correct_k = correct.view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res, pred


def adjust_learning_rate(optimizer, epoch, learning_rate, end_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in [round(end_epoch * 0.333), round(end_epoch * 0.666)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.6
        learning_rate = learning_rate * 0.6
        print('Adjust_learning_rate ' + str(epoch))
        print('New_LearningRate: {}'.format(learning_rate))


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



