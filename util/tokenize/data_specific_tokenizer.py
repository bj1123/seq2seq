from util.tokenize.base_tokenizer import *
import pandas as pd


class MTSpaceTokenizer(SpaceTokenizer):
    def __init__(self, dir_path, prefix, vocab_size, use_sos=True, use_eos=True, **kwargs):
        super(MTSpaceTokenizer, self).__init__(dir_path, prefix, vocab_size, use_sos, use_eos, **kwargs)

    def _read_file(self, file_path, **kwargs):
        f = open(file_path, 'r', encoding='utf-8')
        return f.readlines()

    def _encode_file(self, inp, out, **kwargs):
        res = self._read_file(inp, **kwargs)
        encoded = [self.encode(i) for i in res]
        df = pd.DataFrame({'texts': encoded})
        df.to_pickle(out)


class WikiLargeTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size, tokenizer_class=tokenizers.BertWordPieceTokenizer,
                 morph_analyzer_class=NullAnalyzer, cleanser_class=NullCleanser,
                 use_imap=True, split_jamo=False, **kwargs):
        super(WikiLargeTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                                 tokenizer_class=tokenizer_class,
                                                 morph_analyzer_class=morph_analyzer_class,
                                                 cleanser_class=cleanser_class,
                                                 use_imap=use_imap,
                                                 split_jamo=split_jamo,
                                                 **kwargs)

    def _read_file(self, file_path, **kwargs):
        with open(file_path, 'r', encoding='utf-8') as f:
            res = f.readlines()
            res = list(map(lambda x: x, res))
        return res

    def _encode_file(self, inp, out, **kwargs):
        res = self._read_file(inp, **kwargs)
        encoded = [self.tokenizer.encode(i.rstrip()).ids for i in res]
        df = pd.DataFrame({'texts': encoded})
        return df


class XLSXMultiTaskTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size, tokenizer_class=tokenizers.BertWordPieceTokenizer,
                 morph_analyzer_class=NullAnalyzer, cleanser_class=NullCleanser, tokens_to_add=None,
                 use_imap=True, split_jamo=False, **kwargs):
        super(XLSXMultiTaskTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                                     tokenizer_class=tokenizer_class,
                                                     morph_analyzer_class=morph_analyzer_class,
                                                     cleanser_class=cleanser_class,
                                                     tokens_to_add=tokens_to_add,
                                                     use_imap=use_imap,
                                                     split_jamo=split_jamo,
                                                     **kwargs)

    @staticmethod
    def read_pickle(file_path):
        res = pd.read_pickle(file_path)
        src = res['src'].tolist()
        tgt = res['tgt'].tolist()
        return src, tgt

    @staticmethod
    def is_korean(texts):
        korean = re.compile('[ㄱ-ㅣ가-힣]+')
        res1 = korean.search(texts[0])
        res2 = korean.search(texts[-1])
        return not (res1 == res2)

    @staticmethod
    def count_words(texts):
        return len(texts)  # criteria: n_sentences
        # return sum([len(i.split()) for i in texts])  # criteria: n_words

    def _read_file(self, file_path, **kwargs):
        ratio = kwargs.get('ratio')  # en / ko
        if os.path.splitext(file_path)[-1] == '.pkl':
            res = []
            src, tgt = self.read_pickle(file_path)
            for i in [src, tgt]:
                iskorean = self.is_korean(src)
                if iskorean:
                    res.extend(i * (int(ratio) - 1))
                    remainder = int(len(src) * (ratio - int(ratio)))
                    res.extend(i[:remainder])
                else:
                    res.extend(i)
            return res
        else:
            return []

    def _encode_file(self, inp, out, **kwargs):
        if os.path.splitext(inp)[-1] == '.pkl':
            src, tgt = self.read_pickle(inp)
            src_encoded = [self.tokenizer.encode(i.rstrip()).ids for i in src]
            tgt_encoded = [self.tokenizer.encode(i.rstrip()).ids for i in tgt]
            df = pd.DataFrame({'src': src_encoded, 'tgt': tgt_encoded})
            return df

    def language_ratio(self, file_path):
        ko = 0
        en = 0
        files = self._get_files(file_path)
        for file in files:
            if os.path.splitext(file)[-1] == '.pkl':
                src, tgt = self.read_pickle(file)
                for i in [src, tgt]:
                    if self.is_korean(i):
                        ko += self.count_words(i)
                    else:
                        en += self.count_words(i)
        return en / ko

    def _learn_tokenizer(self, file_path, **kwargs):
        full_path = os.path.join(self.directory_path, file_path)
        ratio = self.language_ratio(full_path)
        new_kwargs = dict(kwargs, ratio=ratio)
        enc = super()._learn_tokenizer(file_path, **new_kwargs)
        return enc



