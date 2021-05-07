from util.tokenize.base_tokenizer import *
from util.tokenize.index_mapper import *
import pandas as pd
import re


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
        self._save_df(df, out)


class WikiLargeTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size=10000, tokenizer_class='wp',
                 morph_analyzer_class=NullAnalyzer, cleanser_class=NullCleanser,
                 imap_type=IMap, split_jamo=False, **kwargs):
        super(WikiLargeTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                                 tokenizer_class=tokenizer_class,
                                                 morph_analyzer_class=morph_analyzer_class,
                                                 cleanser_class=cleanser_class,
                                                 imap_type=imap_type,
                                                 split_jamo=split_jamo,
                                                 **kwargs)

    def _read_file(self, file_path, **kwargs):
        with open(file_path, 'r', encoding='utf-8') as f:
            res = f.readlines()
            res = list(map(lambda x: x, res))
        return res

    def _encode_file(self, inp, out, **kwargs):
        res = self._read_file(inp, **kwargs)
        # encoded = [self.tokenizer.encode(i.rstrip()).ids for i in res]
        encoded = [self.tokenizer.encode(self.morph_analyzer.to_morphs(i.rstrip())).ids for i in res]
        df = pd.DataFrame({'texts': encoded})
        self._save_df(df, out)


class MNMTTargetLangTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size=10000, tokenizer_class='wp',
                 morph_analyzer_class=NullAnalyzer, cleanser_class=NullCleanser,
                 imap_type=IMap, split_jamo=False, target_lang=None **kwargs):
        super(MNMTTargetLangTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                                 tokenizer_class=tokenizer_class,
                                                 morph_analyzer_class=morph_analyzer_class,
                                                 cleanser_class=cleanser_class,
                                                 imap_type=imap_type,
                                                 split_jamo=split_jamo,
                                                 **kwargs)
        self.target_lang = target_lang

    @staticmethod
    def get_language(filepath):
        return os.path.basename(os.path.dirname(filepath))

    def _read_file(self, file_path, **kwargs):
        lang = self.get_language(file_path)
        if lang == self.target_lang:
            with open(file_path, 'r', encoding='utf-8') as f:
                res = f.readlines()
                res = list(map(lambda x: x, res))
            return res

    def _encode_file(self, inp, out, **kwargs):
        res = self._read_file(inp, **kwargs)
        if res:
            # encoded = [self.tokenizer.encode(i.rstrip()).ids for i in res]
            encoded = [self.tokenizer.encode(self.morph_analyzer.to_morphs(i.rstrip())).ids for i in res]
            df = pd.DataFrame({'texts': encoded})
            self._save_df(df, out)


class MultiTaskTokenizer(HFTokenizer):  # for un-corpus
    def __init__(self, directory_path, prefix, vocab_size=30000, tokenizer_class='wp',
                 morph_analyzer_class=NullAnalyzer, cleanser_class=NullCleanser, tokens_to_add=None,
                 imap_type=IMap, split_jamo=False, use_control_token=True, **kwargs):
        from util.tokenize.data_reformatter import MultitaskReformatter
        super(MultiTaskTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                                 tokenizer_class=tokenizer_class,
                                                 morph_analyzer_class=morph_analyzer_class,
                                                 cleanser_class=cleanser_class,
                                                 tokens_to_add=tokens_to_add,
                                                 imap_type=imap_type,
                                                 split_jamo=split_jamo,
                                                 **kwargs)
        self.task_map = MultitaskReformatter.tasks_map
        self.use_control_token = use_control_token
        if self.use_control_token:
            self.control_tokens = set()

    @staticmethod
    def read_pickle(file_path):
        res = pd.read_pickle(file_path)
        src = res['src'].tolist()
        tgt = res['tgt'].tolist()
        return src, tgt, MultiTaskTokenizer.is_simplification(file_path)

    @staticmethod
    def is_simplification(filepath):
        return 'simplification' in os.path.basename(os.path.dirname(filepath))

    @staticmethod
    def is_korean(texts):
        korean = re.compile('[ㄱ-ㅣ가-힣]+')
        if not texts[0]:
            return False
        res1 = korean.search(texts[0])
        res2 = korean.search(texts[-1])
        return not (res1 == res2)

    @staticmethod
    def count_words(texts):
        if not texts[0]:
            return 0
        return len(texts)  # criteria: n_sentences
        # return sum([len(i.split()) for i in texts])  # criteria: n_words

    @staticmethod
    def get_control_tokens(text):
        splited = text.split()
        return splited[:3]

    def update_control_tokens(self, text):
        tokens = self.get_control_tokens(text)
        self.control_tokens.update(tokens)

    def _read_file(self, file_path, **kwargs):
        ratio = kwargs.get('ratio')  # en / ko
        if os.path.splitext(file_path)[-1] == '.pkl':
            res = []
            src, tgt, is_simplification = self.read_pickle(file_path)
            for i in [src, tgt]:
                if i[0]:
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

    def append_special_tokens(self, texts, base_dir):
        indice = []
        lang = '[KOREAN]' if self.is_korean(texts) else '[ENGLISH]'
        indice.append(self.tokenizer.token_to_id(lang))
        task = self.task_map[base_dir]
        indice.append(self.tokenizer.token_to_id(task))
        return lang, indice

    def access_src_encode(self, indice, text):
        control_tokens, raw_texts = text.split()[:3], ' '.join(text.split()[3:])
        ct_indice = [self.tokenizer.token_to_id(i) for i in control_tokens]
        assert None not in ct_indice
        return indice + ct_indice + self.tokenizer.encode(raw_texts.rstrip()).ids

    def _encode_file(self, inp, out, **kwargs):
        if os.path.splitext(inp)[-1] == '.pkl':
            base_dir = os.path.basename(os.path.dirname(inp))
            src, tgt, _ = self.read_pickle(inp)
            encoded = []
            langs = []
            for idx, texts in enumerate([src, tgt]):
                lang, indice = self.append_special_tokens(texts, base_dir)
                if 'simplification' in base_dir and idx == 0:
                    i_encoded = [self.access_src_encode(indice, i) for i in texts]
                else:
                    i_encoded = [indice + self.tokenizer.encode(i.rstrip()).ids for i in texts]
                langs.append(lang)
                encoded.append(i_encoded)
            target_language = [langs[1]] * len(tgt)
            source_language = [langs[0]] * len(tgt)
            df = pd.DataFrame({'src': encoded[0], 'tgt': encoded[1], 'target_language': target_language,
                               'source_language': source_language})
            return df

    def pre_compute(self, file_path):
        ko = 0
        en = 0
        files = self._get_files(file_path)
        for file in files:
            if os.path.splitext(file)[-1] == '.pkl':
                src, tgt, is_simplification = self.read_pickle(file)
                if is_simplification and self.use_control_token:
                    [self.update_control_tokens(i) for i in src]
                for i in [src, tgt]:
                    if self.is_korean(i):
                        ko += self.count_words(i)
                    else:
                        en += self.count_words(i)
        return en / ko

    def _learn_tokenizer(self, file_path, **kwargs):
        ratio = self.pre_compute(file_path)
        new_kwargs = dict(kwargs, ratio=ratio)
        if self.use_control_token:
            self.tokens_to_add += (list(self.control_tokens))
        enc = super()._learn_tokenizer(file_path, **new_kwargs)
        return enc


class UNTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size=30000, tokenizer_class='wp',
                 morph_analyzer_class=NullAnalyzer, cleanser_class=NullCleanser, tokens_to_add=None,
                 imap_type=IMap, split_jamo=False, use_control_token=False, target_lang=None, **kwargs):
        super(UNTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                          tokenizer_class=tokenizer_class,
                                          morph_analyzer_class=morph_analyzer_class,
                                          cleanser_class=cleanser_class,
                                          tokens_to_add=tokens_to_add,
                                          imap_type=imap_type,
                                          split_jamo=split_jamo,
                                          **kwargs)
        self.use_control_token = use_control_token
        self.target_lang = target_lang

    @staticmethod
    def language_token(token):
        return f'[{token.upper()}]'

    @staticmethod
    def get_languages(file_path):
        s = set()
        files = get_files(file_path)
        for file in files:
            dirname = os.path.basename(os.path.dirname(file))
            languages = dirname.split('-')
            for lang in languages:
                s.add(UNTokenizer.language_token(lang))
        return s

    def _read_file(self, file_path, **kwargs):
        df = pd.read_feather(file_path)
        res = []
        for i in df.keys():
            texts = [j + ' \n' for j in df[i].tolist()]
            res.extend(texts)
        return res

    def _encode_file(self, inp, out, **kwargs):
        df = pd.read_feather(inp)
        langs_str = df.keys()[0], df.keys()[1]
        src, tgt = df[langs_str[0]].tolist(), df[langs_str[1]].tolist()
        encoded = []
        for idx, texts in enumerate([src, tgt]):
            if self.use_control_token:
                indice = [self.tokenizer.token_to_id(self.language_token(langs_str[idx]))]
            else:
                indice = []

            batch_encoded = self.tokenizer.encode_batch(texts)
            i_encoded = [indice + i.ids for i in batch_encoded]
            encoded.append(i_encoded)

        src_df = pd.DataFrame({'texts': encoded[0]})
        tgt_df = pd.DataFrame({'texts': encoded[1]})
        src_path = os.path.join(os.path.dirname(out), langs_str[0], os.path.basename(out))
        tgt_path = os.path.join(os.path.dirname(out), langs_str[1], os.path.basename(out))
        self._save_df(src_df, src_path)
        self._save_df(tgt_df, tgt_path)

    def _learn_tokenizer(self, file_path, **kwargs):
        if self.use_control_token:
            self.control_tokens = self.get_languages(file_path)
            if self.tokens_to_add is None:
                self.tokens_to_add = list(self.control_tokens)
            else:
                self.tokens_to_add += list(self.control_tokens)
        new_kwargs = dict(kwargs, filter_words=self.target_lang)
        enc = super()._learn_tokenizer(file_path, **new_kwargs)
        return enc

    def corpus_encode(self, inp_path, **kwargs):
        new_kwargs = dict(kwargs, filter_words=self.target_lang)
        super().corpus_encode(inp_path, **new_kwargs)


class MultilingualTokenizer(HFTokenizer):
    def __init__(self, directory_path, prefix, vocab_size=30000, tokenizer_class='wp',
                 cleanser_class=NullCleanser, tokens_to_add=None,
                 imap_type=IMap, split_jamo=False, target_lang=None, **kwargs):
        kwargs.pop('morph_analyzer_class', None)
        super(MultilingualTokenizer, self).__init__(directory_path, prefix, vocab_size,
                                                    tokenizer_class=tokenizer_class,
                                                    morph_analyzer_class=MultilingualAnalyzer,
                                                    cleanser_class=cleanser_class,
                                                    tokens_to_add=tokens_to_add,
                                                    imap_type=imap_type,
                                                    split_jamo=split_jamo,
                                                    **kwargs)
        self.target_lang = target_lang

    @staticmethod
    def language_token(token):
        return f'[{token.upper()}]'

    @staticmethod
    def get_language(filepath):
        return os.path.basename(os.path.dirname(filepath))

    @staticmethod
    def get_languages_set(file_path):
        s = set()
        files = get_files(file_path)
        for file in files:
            lang = MultilingualTokenizer.get_language(file)
            s.add(MultilingualTokenizer.language_token(lang))
        return s

    def _write_one(self, inp, out, **kwargs):
        lang = self.get_language(inp)
        super()._write_one(inp, out, lang=lang)

    def _read_file(self, file_path, **kwargs):
        lang = self.get_language(file_path)
        if lang == self.target_lang:
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            res = f.readlines()
        if lang == 'ko':
            res = [i for idx, i in enumerate(res) if idx % 2>= 1]  # hard-coded should be modified
        return res

    def _encode_file(self, inp, out, **kwargs):
        with open(inp, 'r', encoding='utf-8') as f:
            res = f.readlines()
        lang = self.get_language(inp)
        if lang == self.target_lang:
            return
        lang_tok = [self.tokenizer.token_to_id(self.language_token(lang))]
        pre_prossed = [self.morph_analyzer.to_morphs(i.rstrip(), lang=lang) for i in res]
        batch_encoded = self.tokenizer.encode_batch(pre_prossed)
        del res
        encoded = [lang_tok + i.ids for i in batch_encoded]

        df = pd.DataFrame({'texts': encoded})
        self._save_df(df, out)

    def _learn_tokenizer(self, file_path, **kwargs):
        self.control_tokens = self.get_languages_set(file_path)
        if self.tokens_to_add is None:
            self.tokens_to_add = list(self.control_tokens)
        else:
            self.tokens_to_add += list(self.control_tokens)
        new_kwargs = dict(kwargs, filter_words=self.target_lang)
        enc = super()._learn_tokenizer(file_path, **new_kwargs)
        return enc

    def corpus_encode(self, inp_path, **kwargs):
        new_kwargs = dict(kwargs, filter_words=self.target_lang)
        super().corpus_encode(inp_path, **new_kwargs)

