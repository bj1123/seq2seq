import nltk
import numpy as np
import argparse
import pandas as pd
from util.files import load_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", type=str)
    parser.add_argument("--sample-path", type=str)
    return parser.parse_args()


def compute_bleu(gts, samples):
    res = []
    for gt, sample in zip(gts, samples):
        res.append(nltk.translate.bleu([gt[1:-1]], sample[:-1]))
    return np.mean(res)


def main():
    args = get_args()
    df = pd.read_pickle(args.gt_path)
    samples = load_json(args.sample_path)
    bleu = compute_bleu(df.texts.tolist(), samples)
    return bleu


if __name__ == '__main__':
    print(main())