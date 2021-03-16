import pandas as pd
import random
import json
import numpy as np
import torch
import os
import pickle
from abc import *
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader
from torchtext import data
from multiprocessing import Process, Manager
from util.files import maybe_read


def truncate(inp, maxlen):
    if len(inp) < maxlen:
        return inp
    else:
        return inp[:maxlen]


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src_texts))
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt_texts))
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def null_count(new, count, sofar):
    return count


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
        ind = [i * bs for i in range(num_buckets)]
        if self.epoch_shuffle:
            random.shuffle(ind)
        return ind

    def sort(self, df):
        return df.sort_values(self.criteria).reset_index(drop=True)

    def to_iterator(self):
        return DataLoader(self, self.size, collate_fn=self.collate_fn)


class TorchTextMTBase(data.Dataset, ABC):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src_texts), len(ex.tgt_texts))

    def __init__(self, batch_size, padding_index, maxlen=512, batch_criteria='token',
                 epoch_shuffle=True, sampling_mode=False, device='cuda', **kwargs):
        self.fields = [('src_texts', data.Field(use_vocab=False, pad_token=padding_index, batch_first=True)),
                  ('tgt_texts', data.Field(use_vocab=False, pad_token=padding_index, batch_first=True)),
                  ('src_len', data.Field(sequential=False, use_vocab=False)),
                  ('tgt_len', data.Field(sequential=False, use_vocab=False))]
        self.padding_index = padding_index
        self.batch_size = batch_size
        self.batch_criteria = batch_criteria
        self.epoch_shuffle = epoch_shuffle
        self.sampling_mode = sampling_mode
        self.device = device
        self.maxlen = maxlen
        self.examples = self.make_examples()
        super(TorchTextMTBase, self).__init__(self.examples, self.fields, **kwargs)
        batch_size_fn = max_tok_len if batch_criteria == 'token' else null_count
        self.ds = data.BucketIterator(sort_within_batch=epoch_shuffle, shuffle=epoch_shuffle, dataset=self, device=device,
                                      batch_size=batch_size, batch_size_fn=batch_size_fn, sort_key=self.sort_key)

    @abstractmethod
    def make_examples(self):
        pass

    def examplify(self, src, tgt):
        def partial_examplify(partial_src, partial_tgt):
            def to_list(idx):
                src = truncate(partial_src.iloc[idx].texts, self.maxlen)
                tgt = truncate(partial_tgt.iloc[idx].texts, self.maxlen)
                return [src, tgt, len(src), len(tgt)]

            examples = [data.Example.fromlist(to_list(i), self.fields) for i in range(len(partial_src))]
            l.extend(examples)

        with Manager() as manager:
            l = manager.list()
            procs = []
            num_thread = 16
            data_per_thread = len(src) // num_thread
            for i in range(num_thread + 1):
                proc = Process(target=partial_examplify, args=(src.iloc[i * data_per_thread:(i + 1) * data_per_thread],
                                                               tgt.iloc[i * data_per_thread:(i + 1) * data_per_thread]))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
            examples = list(l)
        return examples

    def batch_per_epoch(self):
        if self.batch_criteria == 'token':
            avg = sum([i.tgt_len for i in self.examples]) / len(self.examples)
            avg_batch = self.batch_size / avg
            return int(len(self.examples) / avg_batch)
        else:
            return int(len(self.examples) / self.batch_size)

    def iterator(self):
        def reformat(batch):
            return {'src': batch.src_texts,
                    'src_len': batch.src_len,
                    'tgt': batch.tgt_texts[:, :-1],
                    'tgt_len': batch.tgt_len - 1,
                    'label': batch.tgt_texts[:, 1:]}

        self.ds.create_batches()
        for sample in self.ds:
            yield reformat(sample)

    def to_iterator(self):
        return self.iterator()


class TorchTextMTMono(TorchTextMTBase):
    def __init__(self, src_filepath, tgt_filepath, batch_size, padding_index, maxlen=512, batch_criteria='token',
                 epoch_shuffle=True, sampling_mode=False, device='cuda', **kwargs):
        self.fields = [('src_texts', data.Field(use_vocab=False, pad_token=padding_index, batch_first=True)),
                  ('tgt_texts', data.Field(use_vocab=False, pad_token=padding_index, batch_first=True)),
                  ('src_len', data.Field(sequential=False, use_vocab=False)),
                  ('tgt_len', data.Field(sequential=False, use_vocab=False))]
        self.src = maybe_read(src_filepath[0])  # temporary implementation.
        self.tgt = maybe_read(tgt_filepath[0])
        super(TorchTextMTMono, self).__init__(batch_size, padding_index, maxlen, batch_criteria, epoch_shuffle,
                                              sampling_mode, device, **kwargs)

    def make_examples(self):
        return self.examplify(self.src, self.tgt)


class TorchTextMTMulti(TorchTextMTBase):
    def __init__(self, file_path, batch_size, padding_index, maxlen=512, batch_criteria='token',
                 epoch_shuffle=True, sampling_mode=False, device='cuda', **kwargs):
        self.fields = [('src_texts', data.Field(use_vocab=False, pad_token=padding_index, batch_first=True)),
                  ('tgt_texts', data.Field(use_vocab=False, pad_token=padding_index, batch_first=True)),
                  ('src_len', data.Field(sequential=False, use_vocab=False)),
                  ('tgt_len', data.Field(sequential=False, use_vocab=False))]
        self.file_path = file_path
        super(TorchTextMTMulti, self).__init__(batch_size, padding_index, maxlen, batch_criteria, epoch_shuffle,
                                              sampling_mode, device, **kwargs)

    def make_examples(self):
        res = []
        for pair in self.file_path:
            src, tgt = maybe_read(pair[0]), maybe_read(pair[1])
            res.extend(self.examplify(src, tgt))
            res.extend(self.examplify(tgt, src))
        return res


class MTBatchfier(BaseBatchfier):
    def __init__(self, src_filepaths, tgt_filepaths, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'tgt_lens', padding_index=30000, epoch_shuffle=True,
                 sampling_mode=False, device='cuda'):
        super(MTBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle, device)
        self.fl = (src_filepaths, tgt_filepaths)
        self.dfs, self.tot_len, self.eos_idx = self.initialize()
        self.sampling_mode = sampling_mode

    @staticmethod
    def read_file(src, tgt):
        src = maybe_read(src)
        tgt = maybe_read(tgt)
        src_len = [len(i) for i in src.texts]
        tgt_len = [len(i) for i in tgt.texts]
        return pd.DataFrame({'src_texts': src.texts, 'src_lens': src_len,
                             'tgt_texts': tgt.texts, 'tgt_lens': tgt_len})

    def initialize(self):
        l = 0
        dfs = []
        for src, tgt in zip(*self.fl):
            temp = self.read_file(src, tgt)
            l += len(temp)
            dfs.append(temp)
        eos = dfs[-1].tgt_texts[0][-1]
        return dfs, l, eos

    def __len__(self):
        return self.tot_len

    def __iter__(self):
        for df in self.dfs:
            if self.epoch_shuffle:
                df = self.sort(df)
            indice = self.batch_indice(df)
            for l in indice:
                cur_batch = df.iloc[l:l + self.size]
                src_texts = cur_batch['src_texts'].tolist()
                src_lens = cur_batch['src_lens'].tolist()
                tgt_texts = cur_batch['tgt_texts'].tolist()
                tgt_lens = cur_batch['tgt_lens'].tolist()
                for i in range(len(src_texts)):
                    if self.sampling_mode:
                        yield src_texts[i], src_lens[i], tgt_texts[i][:2], 2
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
                'tgt': tgt_texts.to(self.device)[:, :-1],
                'tgt_len': tgt_lens.to(self.device) - 1,
                'label': tgt_texts[:, 1:].to(self.device)}


class MultitaskBatchfier(BaseBatchfier):
    def __init__(self, df_paths, special_token_indice,
                 batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 256,
                 criteria: str = 'tgt_lens', padding_index=30000, epoch_shuffle=True,
                 sampling_mode=False, mask_ratio=0.15, device='cuda'):
        super(MultitaskBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                                 epoch_shuffle, device)
        self.df_paths = df_paths
        self.special_token_indice = special_token_indice  # tokenizer indice
        self.special_token_dic = {i: {j: idx for idx, j in enumerate(special_token_indice[i].keys())}
                                  for i in special_token_indice}  # indice for feeding the model
        print(self.special_token_dic)
        self.mask_ratio = mask_ratio
        self.dfs, self.tot_len, self.eos_idx = self.initialize()
        self.sampling_mode = sampling_mode

    def truncate_line(self, text, length):
        if length > self.maxlen:
            length = self.maxlen
            text = text[:self.maxlen]
        return text, length

    def cat_dfs(self, dfs):
        dfs = dfs + [self.add_autoencoding_data(dfs)]
        n_tokens = [sum(i.tgt_lens.to_list()) for i in dfs]
        # maxl = max([len(i) for i in dfs])  # criteria : n_sentences
        maxl = max(n_tokens)  # criteria : n_tokens
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
        src_len = [len(i) for i in df.src]
        tgt_len = [len(i) for i in df.tgt]

        tgt_language_indice = self.special_token_dic['language'][df.target_language[0]]
        src_language_indice = self.special_token_dic['language'][df.source_language[0]]
        task_indice = [self.special_token_dic['task'][task] for _ in range(len(df))]
        tgt_language_indice = [tgt_language_indice] * len(df)
        src_language_indice = [src_language_indice] * len(df)
        dfs = list()
        dfs.append(pd.DataFrame({'src_texts': df.src, 'src_lens': src_len,
                                 'tgt_texts': df.tgt, 'tgt_lens': tgt_len, 'tasks': task_indice,
                                 'src_languages': src_language_indice,
                                 'tgt_languages': tgt_language_indice}))
        if 'TRANSLATION' in task:
            dfs.append(pd.DataFrame({'src_texts': df.tgt, 'src_lens': tgt_len,
                                     'tgt_texts': df.src, 'tgt_lens': src_len, 'tasks': task_indice,
                                     'src_languages': tgt_language_indice,
                                     'tgt_languages': src_language_indice}))
        return dfs

    def add_autoencoding_data(self, dfs):
        def mask_sentence(text, rate, mask_token_ind):
            text = np.array(text)
            ind = np.random.choice(np.arange(3, len(text)), int(len(text) * rate), replace=False)
            text[ind] = mask_token_ind
            return text.tolist()

        ndf = len(dfs)
        portion = 1 / ndf
        srcs = []
        tgts = []
        src_languages = []
        src_lens = []
        for df in dfs:
            new_df = df.sample(frac=portion)
            for _, row in new_df.iterrows():
                if len(row.src_texts) < 5:
                    src = row.tgt_texts
                    src_len = row.tgt_lens
                    src_lang = row.tgt_languages
                else:
                    i = random.randint(0, 1)
                    if i:
                        src = row.tgt_texts
                        src_len = row.tgt_lens
                        src_lang = row.tgt_languages
                    else:
                        src = row.src_texts
                        src_len = row.src_lens
                        src_lang = row.src_languages
                tgt = mask_sentence(src, self.mask_ratio, self.special_token_indice['symbols']['[MASK]'])
                srcs.append(src)
                tgts.append(tgt)
                src_lens.append(src_len)
                src_languages.append(src_lang)

        return pd.DataFrame({'src_texts': tgts, 'src_lens': src_lens, 'tgt_texts': srcs, 'tgt_lens': src_lens,
                             'tasks': [self.special_token_dic['task']['[TRANSLATION]']] * len(tgts),
                             'src_languages': src_languages, 'tgt_languages': src_languages})

    def initialize(self):
        dfs = []
        for task, df_path in self.df_paths.items():
            for path in df_path:
                df = self.read_df(task, path)
                dfs.extend(df)
        eos = dfs[0].tgt_texts[0][-1]
        ml = max([len(i) for i in dfs]) * (len(dfs) + 1)
        return dfs, ml, eos

    def __len__(self):
        return self.tot_len

    def __iter__(self):
        df = self.cat_dfs(self.dfs)
        if self.epoch_shuffle:
            df = self.sort(df)
        indice = self.batch_indice(df)
        for l in indice:
            cur_batch = df.iloc[l:l + self.size]
            tgt_languages = cur_batch['tgt_languages'].tolist()
            src_texts = cur_batch['src_texts'].tolist()
            src_lens = cur_batch['src_lens'].tolist()
            tgt_texts = cur_batch['tgt_texts'].tolist()
            tgt_lens = cur_batch['tgt_lens'].tolist()
            for i in range(len(src_texts)):
                src_text, src_len = self.truncate_line(src_texts[i], src_lens[i])
                tgt_text, tgt_len = self.truncate_line(tgt_texts[i], tgt_lens[i])
                if self.sampling_mode:
                    yield src_text, src_len, tgt_text[:1], 1, tgt_languages[i]
                else:
                    yield src_text, src_len, tgt_text, tgt_len, tgt_languages[i]

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[3] for item in batch]).long()
        tgt_languages = torch.Tensor([item[4] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts[:, :-1].to(self.device),
                'tgt_len': tgt_lens.to(self.device) - 1,
                'label': tgt_texts[:, 1:].to(self.device),
                'tgt_language': tgt_languages.to(self.device)}


class MultitaskInferBatchfier(BaseBatchfier):  # batchfy from raw text
    def __init__(self, file_path, special_token_indice, tokenizer, batch_size: int = 32, seq_len=512, minlen=50,
                 maxlen: int = 4096,
                 padding_index=30000, device='cuda'):
        print(tokenizer)
        super(MultitaskInferBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen,
                                                      None, padding_index, False, device)
        self.filepath = file_path
        self.special_token_indice = special_token_indice
        self.special_token_dic = {i: {j: idx for idx, j in enumerate(special_token_indice[i].keys())}
                                  for i in special_token_indice}  # indice for feeding the model
        print(self.special_token_dic)
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
            access_control_token = [tokenizer.token_to_id('<WORDRANKRATIO_0.5>'),
                                    tokenizer.token_to_id('<LEVENSHTEIN_0.5>'),
                                    tokenizer.token_to_id('<LENGTHRATIO_1.0>')]
            temp = [tokenizer.token_to_id('[KOREAN]'), tokenizer.token_to_id('[SIMPLIFICATION]')]
            lang = self.special_token_dic['language']['[KOREAN]']
            yield prefix + access_control_token + text, len(text) + len(prefix + access_control_token), \
                  temp + [tokenizer.token_to_id('[SOS]')], 3, lang

    def collate_fn(self, batch):
        src_texts = [torch.Tensor(item[0]).long() for item in batch]
        src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=self.padding_index)
        tgt_texts = [torch.Tensor(item[2]).long() for item in batch]
        tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=self.padding_index)
        src_lens = torch.Tensor([item[1] for item in batch]).long()
        tgt_lens = torch.Tensor([item[3] for item in batch]).long()
        lang = torch.Tensor([item[4] for item in batch]).long()
        return {'src': src_texts.to(self.device),
                'src_len': src_lens.to(self.device),
                'tgt': tgt_texts.to(self.device),
                'tgt_len': tgt_lens.to(self.device),
                'label': tgt_texts[:, 1:].to(self.device),
                'tgt_language': lang.to(self.device)}


class ComplexityControlBatchfier(BaseBatchfier):
    def __init__(self, src_filepaths, tgt_filepaths, rare_index, batch_size: int = 32,
                 seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'tgt_lens', padding_index=30000, epoch_shuffle=True,
                 sampling_mode=False, target_rare_rates=[0.03, 0.07, 0.12, 0.20], device='cuda'):
        super(ComplexityControlBatchfier, self).__init__(batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                                         epoch_shuffle, device)
        self.fl = (src_filepaths, tgt_filepaths)
        self.rare_index = rare_index
        self.target_rare_rates = target_rare_rates
        self.dfs, self.tot_len, self.eos_idx = self.initialize()
        self.sampling_mode = sampling_mode

    def read_pickle(self, src, tgt):
        src = pd.read_pickle(src)
        tgt = pd.read_pickle(tgt)

        src_len = [len(i) for i in src.texts]
        tgt_len = [len(i) for i in tgt.texts]
        tgt_cluster = [self.target_cluster(i) for i in tgt.texts]
        return pd.DataFrame({'src_texts': src.texts, 'src_lens': src_len,
                             'tgt_texts': tgt.texts, 'tgt_lens': tgt_len, 'tgt_clusters': tgt_cluster})

    def target_cluster(self, text):
        def rare_ratio(text, rare_ind):
            text = np.array(text[1:])
            res = text > rare_ind
            return res.sum() / len(res)

        target_rare_rates = self.target_rare_rates + [1.0]
        text_rare_ratio = rare_ratio(text, self.rare_index)
        idx = 0
        temp = target_rare_rates[idx]
        while temp < text_rare_ratio:
            idx += 1
            temp = target_rare_rates[idx]
        return idx

    def initialize(self):
        l = 0
        dfs = []
        for src, tgt in zip(*self.fl):
            temp = self.read_pickle(src, tgt)
            dfs.append(temp)
            l += len(temp)
        eos = dfs[-1].tgt_texts[0][-1]
        return dfs, l, eos

    def __len__(self):
        return self.tot_len

    def __iter__(self):
        for df in self.dfs:
            if self.epoch_shuffle:
                df = self.sort(df)
            indice = self.batch_indice(df)
            for l in indice:
                cur_batch = df.iloc[l:l + self.size]
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


class FairBatchfier:
    def __init__(self, dataset, mode='train', device='cuda'):
        from access.fairseq.base import fairseq_preprocess
        from fairseq.utils import import_user_module
        from fairseq import tasks
        self.preprocessed_dir = fairseq_preprocess(dataset)
        self.device = device
        self.exp_dir = self.prepare_exp_dir()
        self.args = self.fairseq_args(self.preprocessed_dir, exp_dir=self.exp_dir)
        import_user_module(self.args)
        self.task = tasks.setup_task(self.args)
        self.load_dataset_splits(self.task, [mode])
        self.max_positions = (1024, 1024)
        self.padding_index = self.task.tgt_dict.pad_index
        self.epoch_itr = self.task.get_batch_iterator(
            dataset=self.task.dataset(mode),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_positions,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size,
            shard_id=self.args.distributed_rank,
            num_workers=self.args.num_workers,
        )

    @staticmethod
    def prepare_exp_dir():
        from access.fairseq.base import get_fairseq_exp_dir
        import shutil
        exp_dir = get_fairseq_exp_dir()
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True)
        return exp_dir

    @staticmethod
    def fairseq_args(
            preprocessed_dir,
            exp_dir,
            ngpus=None,
            max_tokens=5000,
            arch='fconv_iwslt_de_en',
            pretrained_emb_path=None,
            embeddings_dim=None,
            # Transformer (decoder is the same as encoder for now)
            encoder_embed_dim=512,
            encoder_layers=6,
            encoder_attention_heads=8,
            # encoder_decoder_dim_ratio=1,
            # share_embeddings=True,
            max_epoch=50,
            warmup_updates=None,
            lr=0.1,
            min_lr=1e-9,
            dropout=0.2,
            label_smoothing=0.1,
            lr_scheduler='fixed',
            weight_decay=0.0001,
            criterion='label_smoothed_cross_entropy',
            optimizer='nag',
            validations_before_sari_early_stopping=10,
            fp16=False):
        from pathlib import Path
        import shutil
        from fairseq import options
        exp_dir = Path(exp_dir)
        preprocessed_dir = Path(preprocessed_dir)
        exp_dir.mkdir(exist_ok=True, parents=True)
        # Copy dictionaries to exp_dir for generation
        shutil.copy(preprocessed_dir / 'dict.complex.txt', exp_dir)
        shutil.copy(preprocessed_dir / 'dict.simple.txt', exp_dir)
        train_parser = options.get_training_parser()
        # if share_embeddings:
        #     assert encoder_decoder_dim_ratio == 1
        args = [
            '--task',
            'translation',
            preprocessed_dir,
            '--raw-text',
            '--source-lang',
            'complex',
            '--target-lang',
            'simple',
            '--save-dir',
            os.path.join(exp_dir, 'checkpoints'),
            '--clip-norm',
            0.1,
            '--criterion',
            criterion,
            '--no-epoch-checkpoints',
            '--save-interval-updates',
            5000,  # Validate every n updates
            '--validations-before-sari-early-stopping',
            validations_before_sari_early_stopping,
            '--arch',
            arch,

            # '--decoder-out-embed-dim', int(embeddings_dim * encoder_decoder_dim_ratio),  # Output dim of decoder
            '--max-tokens',
            max_tokens,
            '--max-epoch',
            max_epoch,
            '--lr-scheduler',
            lr_scheduler,
            '--dropout',
            dropout,
            '--lr',
            lr,
            '--lr-shrink',
            0.5,  # For reduce lr on plateau scheduler
            '--min-lr',
            min_lr,
            '--weight-decay',
            weight_decay,
            '--optimizer',
            optimizer,
            '--label-smoothing',
            label_smoothing,
            '--seed',
            random.randint(1, 1000),
            # '--force-anneal', '200',
            # '--distributed-world-size', '1',
        ]
        if arch == 'transformer':
            args.extend([
                '--encoder-embed-dim',
                encoder_embed_dim,
                '--encoder-ffn-embed-dim',
                4 * encoder_embed_dim,
                '--encoder-layers',
                encoder_layers,
                '--encoder-attention-heads',
                encoder_attention_heads,
                '--decoder-layers',
                encoder_layers,
                '--decoder-attention-heads',
                encoder_attention_heads,
            ])
        if pretrained_emb_path is not None:
            args.extend(['--encoder-embed-path', pretrained_emb_path if pretrained_emb_path is not None else ''])
            args.extend(['--decoder-embed-path', pretrained_emb_path if pretrained_emb_path is not None else ''])
        if embeddings_dim is not None:
            args.extend(['--encoder-embed-dim', embeddings_dim])  # Input and output dim of encoder
            args.extend(['--decoder-embed-dim', embeddings_dim])  # Input dim of decoder
        if ngpus is not None:
            args.extend(['--distributed-world-size', ngpus])
        # if share_embeddings:
        #     args.append('--share-input-output-embed')
        if fp16:
            args.append('--fp16')
        if warmup_updates is not None:
            args.extend(['--warmup-updates', warmup_updates])
        args = [str(arg) for arg in args]
        train_args = options.parse_args_and_arch(train_parser, args)
        return train_args

    @staticmethod
    def load_dataset_splits(task, splits):
        import itertools
        for split in splits:
            if split == 'train':
                task.load_dataset(split, combine=True)
            else:
                for k in itertools.count():
                    split_k = split + (str(k) if k > 0 else '')
                    try:
                        task.load_dataset(split_k, combine=False)
                    except FileNotFoundError as e:
                        if k > 0:
                            break
                        raise e

    def __len__(self):
        return len(self.epoch_itr)

    def convert_inp(self, inp):
        new_inp = {}
        net = inp['net_input']
        new_inp['src'] = net['src_tokens'].to(self.device)
        new_inp['src_len'] = net['src_lengths'].to(self.device)
        new_inp['tgt'] = net['prev_output_tokens'].to(self.device)
        new_inp['tgt_len'] = (net['prev_output_tokens'] != 1).sum(dim=1).to(self.device)
        new_inp['label'] = inp['target'].to(self.device)
        return new_inp

    def to_iterator(self):
        itr = self.epoch_itr.next_epoch_itr()
        for inp in itr:
            new_inp = self.convert_inp(inp)
            yield new_inp


class FairTestBatchfier(IterableDataset, FairBatchfier):
    def __init__(self, dataset, batch_size, ratios=[0.75, 0.75, 0.75, 0.95], device='cuda'):
        super(FairTestBatchfier, self).__init__(dataset, 'test', device)
        assert len(ratios) == 4
        print(ratios)
        self.strings = ['<DEPENDENCYTREEDEPTHRATIO_{}>', '<WORDRANKRATIO_{}>', '<LEVENSHTEIN_{}>', '<LENGTHRATIO_{}>']
        self.indice = [self.task.src_dict.index(self.strings[i].format(ratios[i])) for i in range(len(ratios))]
        self.filepath = f'/home/bj1123/access/resources/datasets/{dataset}/fairseq_preprocessed/test.complex-simple.complex'
        self.task.tgt_dict.add_symbol('<eos>')
        self.sos_idx = self.task.tgt_dict.index('</s>')
        self.eos_idx = 2
        # self.eos_idx = self.task.tgt_dict.index('<pad>')
        # self.eos_idx = self.task.tgt_dict.index('▁.')
        self.size = batch_size

    def __iter__(self):
        with open(self.filepath) as f:
            lines = f.readlines()
        for line in lines:
            src_text = self.indice + self.task.src_dict.encode_line(line).tolist()
            src_len = len(src_text)
            tgt_text = [self.sos_idx]
            yield src_text, src_len, tgt_text, 1

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

    def to_iterator(self):
        return DataLoader(self, self.size, collate_fn=self.collate_fn)


def get_fair_batchfier(dataset, device):
    train_batchfier = FairBatchfier(dataset, 'train', device)
    valid_batchfier = FairBatchfier(dataset, 'valid', device)
    return train_batchfier, valid_batchfier
