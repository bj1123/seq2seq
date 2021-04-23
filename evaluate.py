import numpy as np
import argparse
import pandas as pd
from util.files import load_json, maybe_read


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", type=str)
    parser.add_argument("--sample-path", type=str)
    return parser.parse_args()


def tos(l):
    return ' '.join(map(str,l))


def tos2(l, ismulti=False):
    s = 2 if ismulti else 1
    return ' '.join(map(str,l[s:-1]))


def save(l, path):
    with open(path,'w') as f:
        for i in l:
            f.write(i + '\n')


def main():
    args = get_args()
    gt = maybe_read(args.gt_path)
    ismulti = 'semi' in args.sample_path
    print(f'ismulti is : {ismulti}')
    samples = load_json(args.sample_path)
    res = list(map(tos, samples))
    save(res, 'sampled.txt')
    res2 = [tos2(i, ismulti) for i in gt.texts]
    save(res2, 'gt.txt')


if __name__ == '__main__':
    main()