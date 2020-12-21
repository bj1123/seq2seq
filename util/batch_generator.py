import pandas as pd
import random
import json
import numpy as np
import torch
import os
import pickle
from abc import *
from torch.utils.data.dataset import Dataset, IterableDataset


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

    def batch_indice(self, df):
        num_buckets = len(df) // self.size + (len(df) % self.size != 0)
        bs = self.size
        ind = [i*bs for i in range(num_buckets)]
        if self.epoch_shuffle:
            random.shuffle(ind)
        return ind

    def sort(self, df):
        return df.sort_values(self.criteria).reset_index(drop=True)


class MTBatchfier(BaseBatchfier):
    def __init__(self, df_path, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'tgt_lens', padding_index=30000, epoch_shuffle=True,
                 sampling_mode=False, device='cuda'):
        super(MTBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle, device)
        self.df_path = df_path
        self.dfs, self.tot_len, self.eos_idx = self.initialize()
        self.sampling_mode = sampling_mode

    @staticmethod
    def merge_dfs(path):
        df = pd.read_pickle(path)
        src_len = [len(i) for i in df.src]
        tgt_len = [len(i) for i in tgt.tgt]
        return pd.DataFrame({'src_texts': df.src, 'src_lens': src_len,
                             'tgt_texts': df.tgt, 'tgt_lens': tgt_len})

    def initialize(self):
        l = 0
        dfs = []
        for i in self.df_path:
            dfs.append(self.read_pickle(i))
            l += len(temp)
        eos = temp.texts[0][-1]
        return dfs, l, eos

    def __len__(self):
        return self.tot_len

    def __iter__(self):
        for df in self.dfs:
            if self.epoch_shuffle:
                df = self.sort(df)
            indice = self.batch_indice(df)
            for l in indice:
                cur_batch = df.iloc[l:l+self.size]
                src_texts = cur_batch['src_texts'].tolist()
                src_lens = cur_batch['src_lens'].tolist()
                tgt_texts = cur_batch['tgt_texts'].tolist()
                tgt_lens = cur_batch['tgt_lens'].tolist()
                for i in range(len(src_texts)):
                    if self.sampling_mode:
                        yield src_texts[i], src_lens[i], tgt_texts[i][:1], 1
                    else:
                        yield src_texts[i], src_lens[i], tgt_texts[i], tgt_lens[i]

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[3] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts.to(self.device),
                'tgt_len': tgt_lens.to(self.device),
                'label': tgt_texts[:, 1:].to(self.device)}


class MultitaskBatchfier(BaseBatchfier):
    def __init__(self, df_paths, special_token_indice,
                 batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'tgt_lens', padding_index=30000, epoch_shuffle=True,
                 sampling_mode=False, mask_ratio=0.15, device='cuda'):
        super(MultitaskBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle, device)
        self.df_paths = df_paths
        self.special_token_indice = special_token_indice  # tokenizer indice
        self.dfs, self.tot_len, self.eos_idx = self.initialize()
        self.sampling_mode = sampling_mode
        self.mask_ratio = mask_ratio
        self.special_token_dic = {i:{j:idx for idx, j in enumerate(x[i].keys())} for i in special_token_indice}  # indice for feeding the model

    @staticmethod
    def cat_dfs(dfs):
        n_tokens = [sum(i.tgt_lens.to_list()) for i in dfs]
        # maxl = max([len(i) for i in dfs])  # criteria : n_sentences
        maxl = max(n_tokens)  #  criteria : n_tokens
        new_dfs = []
        for ind, i in enumerate(dfs):
            l = len(i)
            # ratio = maxl / l  # criteria : n_sentences
            ratio = maxl / n_tokens[ind]
            i = i.sample(frac=1.0).reset_index(drop=True)
            to_duplicate = ratio
            if to_duplicate > 1:
                dup_ind = int(l * (to_duplicate - int(to_duplicate)))
                i = pd.concat([i] * int(to_duplicate) + [i[:dup_ind]]).reset_index(drop=True)
            new_dfs.append(i)
        return pd.concat(new_dfs).reset_index(drop=True)

    def read_df(self, task, df_path):
        df = pd.read_pickle(df_path)
        src_len = [len(i) + 2 for i in df.src]
        tgt_len = [len(i) + 2 for i in df.tgt]
        st_t = self.special_token_indice

        src_texts = [[st_t['language'][i.source_language], st_t['task'][task]]
                     + i.src for idx, i in df.iterrows()]
        tgt_texts = [[st_t['language'][i.target_language], st_t['task'][task]]
                     + i.tgt for idx, i in df.iterrows()]

        task_indice = [self.special_token_dic['task'][task] for _ in range(len(df))]
        tgt_language_indice = [self.special_token_dic['language'][i] for i in df.target_language]
        src_language_indice = [self.special_token_dic['language'][i] for i in df.source_language]
        dfs = list()
        dfs.append(pd.DataFrame({'src_texts': src_texts, 'src_lens': src_len,
                                 'tgt_texts': tgt_texts, 'tgt_lens': tgt_len, 'tasks': task_indice,
                                 'src_languages':src_language_indice,
                                 'tgt_languages':tgt_language_indice}))
        if 'TRANSLATION' in task:
            dfs.append(pd.DataFrame({'src_texts': tgt_texts, 'src_lens': tgt_len,
                                     'tgt_texts': src_texts, 'tgt_lens': src_len, 'tasks': task_indice,
                                     'src_languages':tgt_language_indice,
                                     'tgt_languages':src_language_indice}))
        return dfs

    def add_autoencoding_data(self, dfs):
        def mask_sentence(text, rate, mask_token_ind):
            text = np.array(text)
            ind = np.random.choice(np.arange(3, len(text)), int(len(text) * rate), replace=False)
            text[ind] = mask_token_ind
            return text.tolist()

        ndf = len(dfs)
        portion = 1 / ndf
        samples = []
        for df in dfs:
            new_df = df.sample(frac=portion)
            for row in new_df:
                if row.src < 3:
                    src = row.tgt
                    src_lang = row.target_language
                else:
                    i = random.randint(0,1)
                    if i:
                        src = row.tgt
                        src_lang = row.target_language
                    else:
                        src = row.src
                        src_lang = row.source_language
                tgt = mask_sentence(src, self.mask_ratio, self.special_token_indice['symbols']['[MASK]'])
                tgt_lang = src_lang
                samples.append(pd.Series({'src_texts':src, 'src_lens':len(src), 'tgt_texts':tgt,
                                          'tgt_lens':len(tgt), 'tasks':self.special_token_dic['task']['[TRANSLATION]'],
                                          'src_languages':src_lang,'tgt_languages':tgt_lang}))
        return pd.DataFrame(samples)

    def initialize(self):
        dfs = []
        for task, df_path in self.df_paths.items():
            for path in df_path:
                df = self.read_df(task, path)
                dfs.extend(df)
        eos = dfs[0].tgt_texts[0][-1]
        dfs.append(self.add_autoencoding_data(dfs))
        ml = max([len(i) for i in dfs]) * len(dfs)
        return dfs, ml, eos

    def __len__(self):
        return self.tot_len

    def __iter__(self):
        df = self.cat_dfs(self.dfs)
        if self.epoch_shuffle:
            df = self.sort(df)
        indice = self.batch_indice(df)
        for l in indice:
            cur_batch = df.iloc[l:l+self.size]
            tasks = cur_batch['tasks'].tolist()
            src_texts = cur_batch['src_texts'].tolist()
            src_lens = cur_batch['src_lens'].tolist()
            tgt_texts = cur_batch['tgt_texts'].tolist()
            tgt_lens = cur_batch['tgt_lens'].tolist()
            for i in range(len(src_texts)):
                if self.sampling_mode:
                    yield src_texts[i], src_lens[i], tgt_texts[i][:1], 1, tasks[i]
                else:
                    yield src_texts[i], src_lens[i], tgt_texts[i], tgt_lens[i], tasks[i]

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[3] for item in batch]).long()
        tasks = torch.Tensor([item[4] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts.to(self.device),
                'tgt_len': tgt_lens.to(self.device),
                'label': tgt_texts[:, 1:].to(self.device),
                'tasks': tasks.to(self.device)}


class MultitaskInferBatchfier(BaseBatchfier): # batchfy from raw text
    def __init__(self, file_path, tokenizer, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 padding_index=30000, device='cuda' ):
        super(MultitaskInferBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen,
                                                      None, padding_index, False, device)
        self.filepath = file_path
        self.tokenizer = tokenizer
        self.texts = self.initialize()
        self.eos_idx = tokenizer.token_to_id('[EOS]')

    def initialize(self):
        with open(self.filepath) as f:
            lines = f.readlines()
        return [self.tokenizer.encode(i) for i in lines]

    def __len__(self):
        return len(self.texts)

    def __iter__(self):
        texts = self.texts
        tokenizer = self.tokenizer
        for text in texts:
            prefix = [tokenizer.token_to_id('[KOREAN]'), tokenizer.token_to_id('[SIMPLIFICATION]')]
            temp = [tokenizer.token_to_id('[KOREAN]'), tokenizer.token_to_id('[SIMPLIFICATION]')]
            yield prefix + text, len(text) + 2, temp + [tokenizer.token_to_id('[SOS]')], 3,\
                  tokenizer.token_to_id('[TRANSLATION]')

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[3] for item in batch]).long()
        tasks = torch.Tensor([item[4] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts.to(self.device),
                'tgt_len': tgt_lens.to(self.device),
                'label': tgt_texts[:, 1:].to(self.device),
                'tasks': tasks.to(self.device)}



class ComplexityControlBatchfier(BaseBatchfier):
    def __init__(self, df_path, probs_path, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'tgt_lens', padding_index=30000, epoch_shuffle=True,
                 sampling_mode=False, rare_probs=[0.9], target_probs = [0.03, 0.07, 0.12, 0.20], device='cuda'):
        super(ComplexityControlBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle, device)
        self.df_path = df_path
        self.probs = json.load(open(probs_path))
        self.dfs, self.tot_len, self.eos_idx = self.initialize()
        self.rare_index = self.get_indices(self.probs, self.rare_probs)[-1]
        self.target_probs = target_probs
        self.sampling_mode = sampling_mode

    def merge_dfs(self, path):
        df = pd.read_pickle(path)
        src_len = [len(i) for i in df.src]
        tgt_len = [len(i) for i in df.tgt]
        tgt_cluster = [self.target_cluster(i) for i in df.tgt]
        return pd.DataFrame({'src_texts': df.src, 'src_lens': src_len,
                             'tgt_texts': df.tgt, 'tgt_lens': tgt_len, 'tgt_clusters':tgt_cluster})

    def target_cluster(self, text):
        def rare_ratio(text, rare_ind):
            text = np.array(text[1:])
            res = text > rare_ind
            return res.sum() / len(res)
        target_probs = self.target_probs + [1.0]
        text_rare_ratio = rare_ratio(text, self.rare_index)
        idx = 0
        temp = rate[idx]
        while temp < text_rare_ratio:
            idx += 1
            temp = rate[idx]
        return idx

    def get_indices(self, cum_prob):
        target_probs = self.target_probs
        cur = 0
        res = []
        for i in target_probs:
            while cum_prob[cur] < i:
                cur += 1
            res.append(cur)
        return res

    def initialize(self):
        l = 0
        dfs = []
        for i in self.df_path:
            dfs.append(self.read_pickle(i))
            l += len(temp)
        eos = temp.texts[0][-1]
        return dfs, l, eos

    def __len__(self):
        return self.tot_len

    def __iter__(self):
        for df in self.dfs:
            if self.epoch_shuffle:
                df = self.sort(df)
            indice = self.batch_indice(df)
            for l in indice:
                cur_batch = df.iloc[l:l+self.size]
                src_texts = cur_batch['src_texts'].tolist()
                src_lens = cur_batch['src_lens'].tolist()
                tgt_texts = cur_batch['tgt_texts'].tolist()
                tgt_lens = cur_batch['tgt_lens'].tolist()
                tgt_clusters = cur_batch['tgt_clusters'].tolist()
                for i in range(len(src_texts)):
                    if self.sampling_mode:
                        yield src_texts[i], src_lens[i], tgt_texts[i][:1], tgt_clusters[i], 1
                    else:
                        yield src_texts[i], src_lens[i], tgt_texts[i], tgt_clusters[i], tgt_lens[i]

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[4] for item in batch]).long()
        tgt_clusters = torch.Tensor([item[3] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts.to(self.device),
                'tgt_len': tgt_lens.to(self.device),
                'tgt_cluster': tgt_clusters.to(self.device),
                'label': tgt_texts[:, 1:].to(self.device)}
