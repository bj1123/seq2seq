import re
from itertools import chain
import collections
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, ngrams, brevity_penalty
from collections import Counter
from fractions import Fraction
import numpy as np
from collections import defaultdict, Counter
from nltk import ngrams


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.
        list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    :param sequence: the source to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def bleu_upto(reference, hypothesis, n_gram):
    res = []
    for i in range(1, n_gram + 1):
        res.append(calc_bleu_ngram(reference, hypothesis, i))
    return res


def calc_bleu_ngram(reference, hypothesis, n_gram):
    score = 0.0
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.tokenize()
        score += sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1)

    return score / len(reference)


def bleu_single(reference, hypothesis, n_gram):
    ratio = 1 / n_gram
    cc = SmoothingFunction()
    return sentence_bleu([reference], hypothesis, (ratio,) * n_gram, cc.method1)


def bleu_multiples(references, hypothesis, n_gram):
    ratio = 1 / n_gram
    score = 0
    cnt = 0
    for i in hypothesis:
        score += sentence_bleu(references, i, (ratio,) * n_gram)
        cnt += 1
    return score / cnt


def count(x, n_gram):
    cnter = collections.Counter()
    for line in x:
        ngram_res = []
        temp = [-1] * (n_gram - 1) + line + [-1] * (n_gram - 1)
        for i in range(len(temp) + n_gram - 1):
            ngram_res.append(str(temp[i:i + n_gram]))
        cnter.update(ngram_res)
    return cnter


def ngram_metrics(token_list, pad=30001):
    if pad in token_list:
        token_list = token_list[:token_list.index(pad)]  # remove possible padding
    stats = defaultdict(float)
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        stats['pct_repeat_%dgrams' % n] = 1.0 - len(counter) / len(ngs)
    return stats


def seq_rep_n(corpus):
    score = 0.0
    total_n = len(corpus)

    for token_list in corpus:
        score += ngram_metrics(token_list)["pct_repeat_4grams"]

    return score / total_n


class Refcnts:
    def __init__(self, references, n):
        self.ref_mcnts = {i: ref_cnts1(references, i) for i in range(1, n + 1)}
        self.ref_lens = [len(i) for i in references]
        self.n = n

    def bleu(self, hypothesis):
        bleu_scores = {i: [] for i in range(1, self.n + 1)}
        for hyp in hypothesis:
            # print(p_denominators,p_numerators)
            p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
            p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
            for i in range(1, self.n + 1):
                p_i = modified_precision(self.ref_mcnts[i], hyp, i)
                # print(p_i)
                p_numerators[i] = p_i.numerator
                p_denominators[i] = p_i.denominator
            hyp_len = len(hyp)
            ref_len = closest_ref_length(iter(self.ref_lens), hyp_len)
            bp = brevity_penalty(ref_len, hyp_len)
            for i in range(1, self.n + 1):
                if p_numerators[i] == 0: p_numerators[i] = 1e-100
                s = (1 / i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1, i + 1))
                s = bp * math.exp(math.fsum(s))
                bleu_scores[i].append(s)

        return [np.mean(bleu_scores[i]) for i in range(1, self.n + 1)]


def build_refcnts(references, n):
    ref_mcnts = {i: ref_cnts1(references, i) for i in range(1, n + 1)}
    ref_lens = [len(i) for i in references]
    return ref_mcnts, ref_lens


def bleu(ref_mcnts, ref_lens, hypothesis, n):
    # print(ref_mcnts)
    # numerator, denominator = 0, 0
    bleu_scores = {i: [] for i in range(1, n + 1)}
    for hyp in hypothesis:
        # print(p_denominators,p_numerators)
        p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
        p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
        for i in range(1, n + 1):
            p_i = modified_precision(ref_mcnts[i], hyp, i)
            # print(p_i)
            p_numerators[i] = p_i.numerator
            p_denominators[i] = p_i.denominator
        hyp_len = len(hyp)
        ref_len = closest_ref_length(iter(ref_lens), hyp_len)
        bp = brevity_penalty(ref_len, hyp_len)
        for i in range(1, n + 1):
            if p_numerators[i] == 0: p_numerators[i] = 1e-100
            s = (1 / i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1, i + 1))
            s = bp * math.exp(math.fsum(s))
            bleu_scores[i].append(s)

    return [np.mean(bleu_scores[i]) for i in range(1, n + 1)]


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def ref_cnts1(references, n):
    ref_mcnts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for i in reference_counts:
            if i not in ref_mcnts:
                ref_mcnts[i] = reference_counts[i]
            elif ref_mcnts[i] < reference_counts[i]:
                ref_mcnts[i] = reference_counts[i]
    return ref_mcnts


def ref_cnts2(references, n):
    ref_mcnts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for i in reference_counts:
            if i not in ref_mcnts:
                ref_mcnts[i] = [reference_counts[i], 0]
            elif ref_mcnts[i][-1] < reference_counts[i]:
                if ref_mcnts[i][0] < reference_counts[i]:
                    ref_mcnts[i] = [reference_counts[i], ref_mcnts[i][0]]
                else:
                    ref_mcnts[i][-1] = reference_counts[i]
    return ref_mcnts


def modified_precision(ref_mcnts, hypothesis, n, isself=False):
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    hyp_mcnts = {}
    for ngram in counts:
        if ngram in ref_mcnts:
            hyp_mcnts[ngram] = ref_mcnts[ngram]
        else:
            hyp_mcnts[ngram] = 0
    if isself:
        clipped_counts = {
            ngram: min(count, ref_mcnts[ngram][1]) if count == ref_mcnts[ngram][0] else min(count, ref_mcnts[ngram][0])
            for ngram, count in counts.items()
        }
    else:
        clipped_counts = {
            ngram: min(count, ref_mcnts.get(ngram, 0)) for ngram, count in counts.items()
        }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def closest_ref_length(ref_lens, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
    )
    return closest_ref_len
