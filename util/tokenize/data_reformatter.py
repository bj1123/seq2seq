import pandas as pd
import os
from util.files import get_files
import shutil


def is_processed(path):
    return 'raw' in os.listdir(path)


class DSReformatter:
    def __init__(self, dir_path, target_format='xlsx'):
        self.dir_path = dir_path
        self.target_format = target_format
        self.filenames = ['wiki.full.aner.ori.test', 'wiki.full.aner.ori.valid', 'wiki.full.aner.ori.train']

    def xlsx_reformat(self, filename):
        dn, bn = os.path.dirname(filename), os.path.basename(filename)
        src_name = filename + '.src'
        tgt_name = filename + '.dst'
        out_name = filename + '.xlsx'
        src_path = os.path.join(self.dir_path,src_name)
        tgt_path = os.path.join(self.dir_path,tgt_name)
        out_path = os.path.join(self.dir_path,out_name)
        if not os.path.exists(out_path):
            with open(src_path, 'r', encoding='utf-8') as f:
                src = f.readlines()
            with open(tgt_path, 'r', encoding='utf-8') as f:
                tgt = f.readlines()
            df = pd.DataFrame({'src': src, 'tgt': tgt})
            df.to_excel(out_path)

        shutil.move(filename, os.path.join(dn, 'raw', bn))

    def start(self, ):
        if is_processed(self.dir_path):
            return
        if self.target_format == 'xlsx':
            for fn in self.filenames:
                self.xlsx_reformat(fn)


class AIHubSplitter:
    def __init__(self, dir_path, ratios=[0.8, 0.1, 0.1]):
        self.dir_path = dir_path
        self.ratios = ratios

    def split(self):
        if is_processed(self.dir_path):
            return
        fn = get_files(self.dir_path)
        for i in fn:
            filename, ext = os.path.splitext(i)
            dn, bn = os.path.dirname(i), os.path.basename(i)

            res = pd.read_excel(i)
            src = res['원문'].tolist()
            tgt = res['번역문'].tolist()
            ti, vi = self.get_indice(len(res))

            # train
            train_name = filename + '_train' + ext
            tr_df = pd.DataFrame({'src': src[:ti], 'tgt': tgt[:ti]})
            tr_df.to_excel(train_name)

            # valid
            test_name = filename + '_test' + ext
            tr_df = pd.DataFrame({'src': src[ti:vi], 'tgt': tgt[ti:vi]})
            tr_df.to_excel(test_name)

            # test
            valid_name = filename + '_valid' + ext
            tr_df = pd.DataFrame({'src': src[vi:], 'tgt': tgt[vi:]})
            tr_df.to_excel(valid_name)
            shutil.move(i, os.path.join(dn,'raw',bn))

    def get_indice(self, data_size):
        train_idx, valid_idx = int(data_size * self.ratios[0]),  int(data_size * (self.ratios[0] + self.ratios[1]))
        return train_idx, valid_idx



