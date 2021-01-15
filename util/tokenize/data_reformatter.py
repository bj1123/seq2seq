import pandas as pd
import os
from util.files import get_files
import shutil


def is_processed(path):
    return 'raw' in os.listdir(path)


class DSReformatter:
    def __init__(self, dir_path, target_format='pickle'):
        self.dir_path = dir_path
        self.target_format = target_format
        self.filenames = get_files(dir_path)
        self.targetnames = ['test', 'train', 'valid']

    def pickle_reformat(self, filename):
        src_name = filename + '.complex'
        tgt_name = filename + '.simple'
        src_path = list(filter(lambda x: src_name in x, self.filenames))[0]
        tgt_path = list(filter(lambda x: tgt_name in x, self.filenames))[0]
        out_name = filename + '.pkl'
        out_path = os.path.join(self.dir_path,out_name)
        if not os.path.exists(out_path):
            with open(src_path, 'r', encoding='utf-8') as f:
                src = f.readlines()
            with open(tgt_path, 'r', encoding='utf-8') as f:
                tgt = f.readlines()
            df = pd.DataFrame({'src': src, 'tgt': tgt})
            df.to_pickle(out_path)

        if not os.path.exists(os.path.join(self.dir_path, 'raw')):
            os.makedirs(os.path.join(self.dir_path, 'raw'))
        shutil.move(src_path, os.path.join(self.dir_path, 'raw', os.path.basename(src_path)))
        shutil.move(tgt_path, os.path.join(self.dir_path, 'raw',  os.path.basename(tgt_path)))

    def start(self):
        if is_processed(self.dir_path):
            return
        if self.target_format == 'pickle':
            for fn in self.targetnames:
                self.pickle_reformat(fn)
        else:
            raise NotImplementedError


class WikiReformatter:
    def __init__(self, dir_path, tokens_per_chunk = 128):
        self.dir_path = dir_path
        self.tokens_per_chunk = tokens_per_chunk

    def start(self):
        if is_processed(self.dir_path):
            return

        fn = get_files(self.dir_path)
        for i in fn:
            filename, ext = os.path.splitext(i)
            dn, bn = os.path.dirname(i), os.path.basename(i)
            out_path = filename + '.pkl'
            if not os.path.exists(os.path.join(dn, 'raw')):
                os.makedirs(os.path.join(dn, 'raw'))
            if not os.path.exists(out_path):
                with open(i, 'r', encoding='utf-8') as f:
                    res = f.readlines()
                tokens = ' '.join([i.strip() for i in res if i.strip()]).split()
                l = len(tokens) // self.tokens_per_chunk
                tgt = [' '.join(tokens[k* self.tokens_per_chunk :(k + 1) * self.tokens_per_chunk]) for k in range(l)]
                df = pd.DataFrame({'src': ['' for _ in range(len(tgt))], 'tgt': tgt})
                df.to_pickle(out_path)

            if not os.path.exists(os.path.join(self.dir_path, 'raw')):
                os.makedirs(os.path.join(self.dir_path, 'raw'))
            shutil.move(i, os.path.join(self.dir_path, 'raw', bn))


class NaverNewsReformatter:
    def __init__(self, dir_path, ratios=[0.8, 0.1, 0.1], tokens_per_chunk=128):
        self.dir_path = dir_path
        self.ratios = ratios
        self.tokens_per_chunk = tokens_per_chunk

    def divide_text(self, text):
        splited = text.split()
        if len(splited) < self.tokens_per_chunk:
            return [text]
        else:
            l = len(splited) // self.tokens_per_chunk
            res = [' '.join(splited[l* self.tokens_per_chunk :(l + 1) * self.tokens_per_chunk]) for l in range(l)]
            return res

    def start(self):
        if is_processed(self.dir_path):
            return
        fn = get_files(self.dir_path)
        train_name = os.path.join(self.dir_path, 'train.pkl')
        test_name = os.path.join(self.dir_path, 'test.pkl')
        valid_name = os.path.join(self.dir_path, 'valid.pkl')
        train_df = None
        test_df = None
        valid_df = None
        for i in fn:
            filename, ext = os.path.splitext(i)
            dn, bn = os.path.dirname(i), os.path.basename(i)
            if not os.path.exists(os.path.join(dn, 'raw')):
                os.makedirs(os.path.join(dn, 'raw'))

            res = pd.read_pickle(i)
            tgt = []
            for text in res['contents'].tolist():
                tgt += self.divide_text(text)
            src = ['' for _ in range(len(tgt))]
            ti, vi = self.get_indice(len(res))

            # train
            tr_df = pd.DataFrame({'src': src[:ti], 'tgt': tgt[:ti]})
            if train_df is None:
                train_df = tr_df
            else:
                train_df = train_df.append(tr_df).reset_index(drop=True)

            # valid
            tr_df = pd.DataFrame({'src': src[ti:vi], 'tgt': tgt[ti:vi]})
            if valid_df is None:
                valid_df = tr_df
            else:
                valid_df = valid_df.append(tr_df).reset_index(drop=True)

            # test
            tr_df = pd.DataFrame({'src': src[vi:], 'tgt': tgt[vi:]})
            if test_df is None:
                test_df = tr_df
            else:
                test_df = test_df.append(tr_df).reset_index(drop=True)
            shutil.move(i, os.path.join(dn, 'raw', bn))

        train_df.to_pickle(train_name)
        test_df.to_pickle(test_name)
        valid_df.to_pickle(valid_name)

    def get_indice(self, data_size):
        train_idx, valid_idx = int(data_size * self.ratios[0]), int(data_size * (self.ratios[0] + self.ratios[1]))
        return train_idx, valid_idx


class AIHubReformatter:
    def __init__(self, dir_path, ratios=[0.8, 0.1, 0.1]):
        self.dir_path = dir_path
        self.ratios = ratios

    def start(self):
        if is_processed(self.dir_path):
            return
        fn = get_files(self.dir_path)
        train_name = os.path.join(self.dir_path, 'train.pkl')
        test_name = os.path.join(self.dir_path, 'test.pkl')
        valid_name = os.path.join(self.dir_path, 'valid.pkl')
        train_df = None
        test_df = None
        valid_df = None
        for i in fn:
            filename, ext = os.path.splitext(i)
            dn, bn = os.path.dirname(i), os.path.basename(i)
            if not os.path.exists(os.path.join(dn,'raw')):
                os.makedirs(os.path.join(dn,'raw'))

            res = pd.read_excel(i)
            src = res['원문'].tolist()
            tgt = res['번역문'].tolist()
            ti, vi = self.get_indice(len(res))

            # train
            tr_df = pd.DataFrame({'src': src[:ti], 'tgt': tgt[:ti]})
            if train_df is None:
                train_df = tr_df
            else:
                train_df = train_df.append(tr_df).reset_index(drop=True)

            # valid
            tr_df = pd.DataFrame({'src': src[ti:vi], 'tgt': tgt[ti:vi]})
            if valid_df is None:
                valid_df = tr_df
            else:
                valid_df = valid_df.append(tr_df).reset_index(drop=True)

            # test
            tr_df = pd.DataFrame({'src': src[vi:], 'tgt': tgt[vi:]})
            if test_df is None:
                test_df = tr_df
            else:
                test_df = test_df.append(tr_df).reset_index(drop=True)
            shutil.move(i, os.path.join(dn,'raw',bn))

        train_df.to_pickle(train_name)
        test_df.to_pickle(test_name)
        valid_df.to_pickle(valid_name)

    def get_indice(self, data_size):
        train_idx, valid_idx = int(data_size * self.ratios[0]),  int(data_size * (self.ratios[0] + self.ratios[1]))
        return train_idx, valid_idx


class MultitaskReformatter:
    reformatter_map = {'aihub_mt':AIHubReformatter, 'simplification':DSReformatter,
                       'naver_news':NaverNewsReformatter, 'wikitext':WikiReformatter}
    tasks_map = {'aihub_mt': '[TRANSLATION]', 'simplification': '[SIMPLIFICATION]', 'naver_news':'[LANGUAGE_MODEL]',
                 'wikitext':'[LANGUAGE_MODEL]'}
    languages = ['[KOREAN]', '[ENGLISH]']

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.tasks = os.listdir(dir_path)  # assume it only includes folder
        self.tokens_to_add = list(set([self.tasks_map[i] for i in self.tasks if i in self.tasks_map])) + self.languages
        self.formatter_classes = {i:self.reformatter_map[i] for i in self.tasks if i in self.reformatter_map}

    def start(self, **kwargs):
        for folder, formatter_class in self.formatter_classes.items():
            path = os.path.join(self.dir_path,folder)
            formatter = formatter_class(path, **kwargs)
            formatter.start()




