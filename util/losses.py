import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
from collections import Counter
import torch.nn as nn
import numpy as np
import math
from abc import ABC


class BaseLoss(_Loss, ABC):
    def __init__(self,):
        super(BaseLoss, self).__init__()
        self.cum_loss = 0
        self.cum_acc = 0

    def clear_loss(self):
        self.cum_loss = 0
        self.cum_acc = 0


class PlainWrapper(BaseLoss):
    def __init__(self):
        raise NotImplementedError
    # To_Do


class LabelSmoothingWrapper(BaseLoss):
    def __init__(self):
        raise NotImplementedError
    # To_Do


class PlainLoss(BaseLoss):
    def __init__(self, padding_idx: int, seq2seq=True):
        super(PlainLoss, self).__init__()
        self.padding_idx = padding_idx
        self.seq2seq = seq2seq
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, y_hat, y):
        if self.seq2seq:
            y_hat = y_hat[:, :-1]
        if len(y_hat.size()) !=2:
            y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
            y = y.view(-1)
        y = y.contiguous()
        l = self.criteria(y_hat,y)
        self.cum_loss +=l
        return l

    def get_description(self, step):
        tok_loss = self.cum_loss
        desc = " token loss : %f, token ppl : %f, acc : %f " % (
            tok_loss / step, math.exp(tok_loss / step), self.cum_acc / step)
        return desc


class LabelSmoothingLoss(BaseLoss):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-1, seq2seq=True, device='cuda'):
        super(LabelSmoothingLoss, self).__init__()
        self.seq2seq = seq2seq
        self.device = device
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing / (vocab_size - 1)
        one_hot = torch.full((vocab_size,), self.smoothing).to(device)
        one_hot[ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.vocab_Size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, y_hat, y):
        if self.seq2seq:
            y_hat = y_hat[:, :-1]
        if len(y_hat.size()) !=2:
            y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
            y = y.view(-1)
        pred = y_hat.log_softmax(dim=-1)

        true_dist = self.one_hot.repeat(y_hat.size(0),1)
        true_dist.scatter_(1, y.data.unsqueeze(1), self.confidence)
        true_dist.masked_fill_((y == self.ignore_index).unsqueeze(1),0)
        l = torch.mean(torch.sum(-true_dist * pred, dim=-1))
        self.cum_loss += l
        return l

    def get_description(self, step):
        tok_loss = self.cum_loss
        desc = " token loss : %f, token ppl : %f, acc : %f " % (
            tok_loss / step, math.exp(tok_loss / step), self.cum_acc / step)
        return desc
