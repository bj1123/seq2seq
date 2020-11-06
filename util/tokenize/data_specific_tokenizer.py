from util.tokenize.base_tokenizer import *


class TempTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size, tokenizer_class=tokenizers.SentencePieceBPETokenizer,
                 morph_analyzer_class=MecabAnalyzer, cleanser_class=NullCleanser,
                 use_imap=True, split_jamo=False, **kwargs
                 ):
        super(TempTokenizer, self).__init__(directory_path, prefix, vocab_size, tokenizer_class,
                                            morph_analyzer_class, cleanser_class, use_imap, split_jamo,
                                            **kwargs)

    def _read_file(self, file_path, **kwargs):
        with open(file_path,'r') as f:
            res = f.readlines()
            res = list(map(lambda x: x.strip(), res))
        return res

    def _encode_file(self, inp, out, **kwargs):
        res = self._read_file(inp, **kwargs)
        encoded = [self.tokenizer.encode(i.rstrip()).ids for i in res]
        df = pd.DataFrame({'texts': encoded})
        df.to_pickle(out)


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