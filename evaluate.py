import numpy as np
import argparse
import pandas as pd
from util.files import load_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", type=str)
    parser.add_argument("--sample-path", type=str)
    return parser.parse_args()


def tos(l):
    return ' '.join(map(str,l))


def tos2(l):
    return ' '.join(map(str,l[1:-1]))


def save(l, path):
    with open(path,'w') as f:
        for i in l:
            f.write(i + '\n')


def main():
    args = get_args()
    gt = pd.read_pickle(args.gt_path)
    samples = load_json(args.sample_path)
    res = list(map(tos, samples))
    save(res, 'sampled.txt')
    res2 = list(map(tos2, gt.texts))
    save(res2, 'gt.txt')


if __name__ == '__main__':
    main()