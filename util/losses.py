import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
from collections import Counter
import torch.nn as nn
import numpy as np
import math
from abc import ABC
# from util.metrics import *
# import pytorch_lightning as pl


class BaseLoss(_Loss, ABC):
    def __init__(self,):
        super(BaseLoss, self).__init__()
        self.cum_loss = 0
        self.cum_acc = 0
        self.cum_cnt = 0
        self.metrics = {}

    def clear_loss(self):
        self.cum_loss = 0
        self.cum_acc = 0
        self.cum_cnt = 0

    def to_log(self, out, inp):  # pytorch_lightning
        d = {}
        for name, l in self.metrics.items():
            res = l(out, inp)
            d[name] = res
        return d


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
        # self.metrics = {'perplexity': Perplexity(padding_idx)}

    def forward(self, out, inp):
        y_hat, y = out['logits'], inp['label']
        if len(y_hat.size()) !=2:
            y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
            y = y.contiguous().view(-1)
        y = y.contiguous()
        l = self.criteria(y_hat,y)
        self.cum_loss +=l
        return l

    def get_description(self):  # for pure pytorch
        tok_loss = self.cum_loss
        desc = " token loss : %f, token ppl : %f, acc : %f " % (
            tok_loss / self.cum_cnt, math.exp(tok_loss / self.cum_cnt), self.cum_acc / self.cum_cnt)
        return desc


class ComplexityLoss(BaseLoss):
    def __init__(self, main_loss:BaseLoss, auxiliary_lambda=1):
        super(ComplexityLoss, self).__init__()
        self.main_loss = main_loss
        self.aux_lambda = auxiliary_lambda
        self.criteria = torch.nn.CrossEntropyLoss()

    def forward(self, out, inp):
        main_loss = self.main_loss(out, inp)
        y_hat, y = out['cluster_logits'], inp['tgt_cluster']
        aux_loss = self.criteria(y_hat,y)
        self.cum_loss += aux_loss
        return main_loss + self.aux_lambda * aux_loss

    def get_description(self, step):
        complexity_loss = self.cum_loss
        tok_loss = self.main_loss.cum_loss
        desc = f"total loss: {(tok_loss + self.aux_lambda * complexity_loss)/step:.3f}" \
               f" token loss : {tok_loss / step:.3f}," \
               f" token ppl : {math.exp(tok_loss / step):.3f} acc : {self.cum_acc / step} "
        return desc

    def clear_loss(self):
        self.main_loss.clear_loss()
        super().clear_loss()


class SentenceAwareLoss(BaseLoss):
    def __init__(self, main_loss: BaseLoss, auxiliary_lambda=1):
        super(SentenceAwareLoss, self).__init__()
        self.main_loss = main_loss
        self.aux_lambda = auxiliary_lambda
        # self.criteria = torch.nn.L1Loss()
        # self.criteria = torch.nn.MSELoss()
        self.criteria = torch.nn.MSELoss(reduction='none')

    def forward(self, out, inp):
        main_loss = self.main_loss(out, inp)
        # return main_loss
        y_hat, y = out['tgt_emb_hat'], out['tgt_emb']
        aux_loss = self.criteria(y_hat, y).mean(-1)
        self.cum_loss += aux_loss.sum()
        self.cum_cnt += aux_loss.numel()
        return main_loss + self.aux_lambda * aux_loss.mean()

    # def get_description(self, step):
    #     tok_loss = self.main_loss.cum_loss
    #     desc = f" token loss : {tok_loss / step:.3f}," \
    #            f" token ppl : {math.exp(tok_loss / step):.3f} acc : {self.cum_acc / step} "
    #     return desc

    def get_description(self):
        token_loss = self.main_loss.cum_loss / self.main_loss.cum_cnt
        mse_loss = self.cum_loss / self.cum_cnt
        # mse_loss = 0
        desc = f"total loss: {token_loss + mse_loss:.3f}" \
               f" token loss : {token_loss:.3f}," \
               f" mse loss : {mse_loss:.3f}," \
               f" token ppl : {math.exp(token_loss):.3f} acc : {self.cum_acc / self.cum_cnt} "
        return desc

    def clear_loss(self):
        self.main_loss.clear_loss()
        super().clear_loss()


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
        # self.metrics = {'perplexity': Perplexity(ignore_index)}

    def forward(self, out, inp):
        y_hat, y = out['logits'], inp['label']
        # if self.seq2seq:
        #     y_hat = y_hat[:, :-1].contiguous()
        if len(y_hat.size()) !=2:
            y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
            y = y.contiguous().view(-1)

        pred = y_hat.log_softmax(dim=-1)

        true_dist = self.one_hot.repeat(y_hat.size(0),1)
        true_dist.scatter_(1, y.data.unsqueeze(1), self.confidence)
        true_dist.masked_fill_((y == self.ignore_index).unsqueeze(1),0)
        tot_loss = torch.sum(-true_dist * pred, dim=-1)
        non_padding_cnt = (y != self.ignore_index).sum()
        l = torch.sum(tot_loss)
        self.cum_loss += l
        self.cum_cnt += non_padding_cnt
        return l / non_padding_cnt

    def get_description(self):
        tok_loss = self.cum_loss
        desc = " token loss : %f, token ppl : %f, acc : %f " % (
            tok_loss / self.cum_cnt, math.exp(tok_loss / self.cum_cnt), self.cum_acc / self.cum_cnt)
        return desc
