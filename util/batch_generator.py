import pandas as pd
import random
import numpy as np
import torch
import os
import pickle
from abc import *
from torch.utils.data.dataset import Dataset, IterableDataset
import math


class BaseBatchfier(IterableDataset):
    def __init__(self, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'lens',
                 padding_index=70000, epoch_shuffle=False, device='cuda'):
        super(BaseBatchfier, self).__init__()
        self.maxlen = maxlen
        self.minlen = minlen
        self.size = batch_size
        self.criteria = criteria
        self.seq_len = seq_len
        self.padding_index = padding_index
        self.epoch_shuffle = epoch_shuffle
        self.device = device
        # self.size = len(self.df) / num_buckets

    def truncate_small(self, df, criteria='lens'):
        lens = np.array(df[criteria])
        indices = np.nonzero((lens < self.minlen).astype(np.int64))[0]
        return df.drop(indices)

    def truncate_large(self, texts, lens):
        new_texts = []
        new_lens = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > self.maxlen:
                new_texts.append(text[:self.maxlen])
                new_lens.append(self.maxlen)
            else:
                remainder = len(text) % self.seq_len
                l = lens[i]
                if remainder and remainder < 10:
                    text = text[:-remainder]
                    l = l - remainder
                new_texts.append(text)
                new_lens.append(l)
        return new_texts, new_lens

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(num_buckets - 1):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)
        random.shuffle(dfs)
        dfs.append(df.iloc[num_buckets - 1 * self.size: num_buckets * self.size])
        df = pd.concat(dfs)
        return df

    def sort(self, df):
        return df.sort_values(self.criteria).reset_index(drop=True)


class MTBatchfier(BaseBatchfier):
    def __init__(self, src_filepath, tgt_filepath, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'tgt_lens',
                 padding_index=70000, epoch_shuffle=False, device='cuda'):
        super(MTBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle, device)
        src_df = pd.read_pickle(src_filepath)
        tgt_df = pd.read_pickle(tgt_filepath)
        self.df = self.merge_dfs(src_df, tgt_df)

    @staticmethod
    def merge_dfs(src, tgt):
        src_len = [len(i) for i in src.texts]
        tgt_len = [len(i) for i in tgt.texts]
        return pd.DataFrame({'src_texts': src.texts, 'src_lens': src_len,
                             'tgt_texts': tgt.texts, 'tgt_lens': tgt_len})

    def __iter__(self):
        df = self.df
        cur_pos = 0
        if self.epoch_shuffle:
            num_buckets = len(df) // self.size + (len(df) % self.size != 0)
            df = self.sort(df)
            df = self.shuffle(df, num_buckets)

        while cur_pos < len(df):
            cur_batch = df.iloc[cur_pos:cur_pos + self.size]
            cur_pos += self.size
            src_texts = cur_batch['src_texts'].to_list()
            src_lens = cur_batch['src_lens'].to_list()
            tgt_texts = cur_batch['tgt_texts'].to_list()
            tgt_lens = cur_batch['tgt_lens'].to_list()
            for i in range(len(tgt_texts)):
                yield src_texts[i], src_lens[i], tgt_texts[i], tgt_lens[i]

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[1] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts.to(self.device),
                'tgt_len': tgt_lens.to(self.device),
                'label': tgt_texts[:, 1:].to(self.device)}
