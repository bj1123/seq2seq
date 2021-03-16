import datasets
import pandas as pd
import os
import argparse
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-path", type=str)
    parser.add_argument("--ratio", type=float, default=0.1)
    return parser.parse_args()


class UNDownloader:
    def __init__(self, dir_path, truncate_ratio=0.1):
        self.language_pairs = ['ar-en', 'ar-es', 'ar-fr', 'ar-ru', 'ar-zh', 'en-es', 'en-fr', 'en-ru', 'en-zh',
                               'es-fr', 'es-ru', 'es-zh', 'fr-ru', 'fr-zh', 'ru-zh']
        self.dir_path = dir_path
        self.ratio = truncate_ratio

    @staticmethod
    def frac(pair, ratio=0.1):
        dataset = datasets.load_dataset('un_pc', pair)
        n_samples = len(dataset['train'])
        tgt_samples = int(n_samples * ratio)
        out = dataset['train'][-tgt_samples:]['translation']
        random.shuffle(out)
        return out

    @staticmethod
    def to_df(dataset):
        languages = list(dataset[0].keys())
        n_samples = len(dataset)
        src = [dataset[i][languages[0]] for i in range(n_samples)]
        tgt = [dataset[i][languages[1]] for i in range(n_samples)]
        df = pd.DataFrame({languages[0]: src, languages[1]: tgt})
        return df

    @staticmethod
    def split(df):
        n = len(df)
        train_idx, valid_idx, test_idx = int(0.8 * n), int(0.9 * n), n
        train = df.iloc[:train_idx].reset_index(drop=True)
        valid = df.iloc[train_idx:valid_idx].reset_index(drop=True)
        test = df.iloc[valid_idx:].reset_index(drop=True)
        return train, valid, test

    def save(self, pair, ratio):
        samples = self.frac(pair, ratio)
        df = self.to_df(samples)
        train, valid, test = self.split(df)
        dir_path = os.path.join(self.dir_path, 'raw', pair)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        train.to_feather(os.path.join(dir_path, 'train.feather'))
        valid.to_feather(os.path.join(dir_path, 'valid.feather'))
        test.to_feather(os.path.join(dir_path, 'test.feather'))

    def run(self):
        for i in self.language_pairs:
            self.save(i, self.ratio)


def main():
    args = get_args()
    downloader = UNDownloader(args.dir_path, args.ratio)
    downloader.run()


if __name__ == '__main__':
    main()