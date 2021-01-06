# =======================================================
#  SARI -- Text Simplification Tunable Evaluation Metric
# =======================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# A Python implementation of the SARI metric for text simplification
# evaluation in the following paper  
#
#     "Optimizing Statistical Machine Translation for Text Simplification"
#     Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and Chris Callison-Burch
#     In Transactions of the Association for Computational Linguistics (TACL) 2015
# 
# There is also a Java implementation of the SARI metric 
# that is integrated into the Joshua MT Decoder. It can 
# be used for tuning Joshua models for a real end-to-end
# text simplification model. 
#

from util.tokenize.data_specific_tokenizer import *
from easse.sari import get_corpus_sari_operation_scores
from util.tokenize import data_specific_tokenizer
from easse.utils.resources import get_orig_sents, get_refs_sents
from util.args import MTArgument
import numpy as np
import re
from collections import Counter
from typing import List
import easse.utils.preprocessing as utils_prep




def corpus_sari(*args, **kwargs):
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(*args, **kwargs)
    return (add_score + keep_score + del_score) / 3


def read_lines(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def get_orig_and_refs_sents(test_set, orig_sents_path=None, refs_sents_paths=None):
    # Get original and reference sentences
    if test_set == "custom":
        assert orig_sents_path is not None
        assert refs_sents_paths is not None
        if type(refs_sents_paths) == str:
            refs_sents_paths = refs_sents_paths.split(",")
        orig_sents = read_lines(orig_sents_path)
        refs_sents = [read_lines(ref_sents_path) for ref_sents_path in refs_sents_paths]
    else:
        orig_sents = get_orig_sents(test_set)
        refs_sents = get_refs_sents(test_set)
    # Final checks
    assert all([len(orig_sents) == len(ref_sents) for ref_sents in refs_sents])
    return orig_sents, refs_sents


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", type=str)
    parser.add_argument("--prob-path", type=str)
    parser.add_argument("--decoder-dir", type=str)
    parser.add_argument("--decoder-prefix", type=str)
    return parser.parse_args()


def get_rare_index(args):
    cum_prob = load_json(args.prob_path)
    return MTArgument.get_indices(cum_prob)[-1]


def rare_ratio(text, rare_ind):
    text = np.array(text[1:])
    res = text > rare_ind
    return res.sum() / len(res)


def replace_parenthesis(text):
    lrb = re.compile(r'\(')
    rrb = re.compile(r'\)')
    text = lrb.sub('-lrb-', text)
    text = rrb.sub('-rrb-', text)
    return text


def main():
    args = get_args()
    rare_ind = get_rare_index(args)
    name = 'turkcorpus_test_legacy'
    decoder = data_specific_tokenizer.WikiLargeTokenizer(args.decoder_dir, args.decoder_prefix)
    system_outs = load_json(args.sample_path)

    ori, refs = get_orig_and_refs_sents(name)
    decoded_system_outs = [decoder.decode(i).lower() for i in system_outs]
    # decoded_system_outs = [replace_parenthesis(i) for i in decoded_system_outs]
    res = get_corpus_sari_operation_scores(ori, decoded_system_outs, refs, tokenizer="13a", lowercase=True, legacy=True,)
    print(res)
    print(sum(res) / 3)

    print(np.mean([rare_ratio(i, rare_ind) for i in system_outs]))


if __name__ == '__main__':
    main()
