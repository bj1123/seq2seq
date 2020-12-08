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
        self.filenames = ['wikilarge.test', 'wikilarge.train', 'wikilarge.valid']

    def pickle_reformat(self, filename):
        src_name = filename + '.complex'
        tgt_name = filename + '.simple'
        out_name = filename + '.pkl'
        src_path = os.path.join(self.dir_path,src_name)
        tgt_path = os.path.join(self.dir_path,tgt_name)
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
        shutil.move(src_path, os.path.join(self.dir_path, 'raw', filename + '.complex'))
        shutil.move(tgt_path, os.path.join(self.dir_path, 'raw', filename + '.simple'))

    def start(self):
        if is_processed(self.dir_path):
            return
        if self.target_format == 'pickle':
            for fn in self.filenames:
                self.pickle_reformat(fn)
        else:
            raise NotImplementedError


class AIHubReformatter:
    def __init__(self, dir_path, ratios=[0.8, 0.1, 0.1]):
        self.dir_path = dir_path
        self.ratios = ratios

    def start(self):
        if is_processed(self.dir_path):
            return
        fn = get_files(self.dir_path)
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
            train_name = filename + '_train.pkl'
            tr_df = pd.DataFrame({'src': src[:ti], 'tgt': tgt[:ti]})
            tr_df.to_pickle(train_name)

            # valid
            test_name = filename + '_test.pkl'
            tr_df = pd.DataFrame({'src': src[ti:vi], 'tgt': tgt[ti:vi]})
            tr_df.to_pickle(test_name)

            # test
            valid_name = filename + '_valid.pkl'
            tr_df = pd.DataFrame({'src': src[vi:], 'tgt': tgt[vi:]})
            tr_df.to_pickle(valid_name)
            shutil.move(i, os.path.join(dn,'raw',bn))

    def get_indice(self, data_size):
        train_idx, valid_idx = int(data_size * self.ratios[0]),  int(data_size * (self.ratios[0] + self.ratios[1]))
        return train_idx, valid_idx


class MultitaskReformatter:
    reformatter_map = {'aihub_mt':AIHubReformatter, 'simplification':DSReformatter}

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.tasks = os.listdir(dir_path)  # assume it only includes folder
        self.formatter_classes = {i:self.reformatter_map[i] for i in self.tasks if i in self.reformatter_map}

    def start(self, **kwargs):
        for folder, formatter_class in self.formatter_classes.items():
            path = os.path.join(self.dir_path,folder)
            formatter = formatter_class(path, **kwargs)
            formatter.start()



