import tokenizers
from util.tokenize.morph_analyzer import *
from multiprocessing import Process
from util.tokenize.cleanser import *
from abc import abstractmethod, ABC
from util.tokenize.index_mapper import *
import collections
import json
from tokenizers import Tokenizer
from tokenizers.models import *
import shutil
import random
from util.files import maybe_read


class BaseTokenizer(ABC):
    def __init__(self, dir_path, prefix, vocab_size=10000, imap_type=IMap, **kwargs):
        self.tokenizer, self.is_trained = self._load_tokenizer(dir_path, prefix)
        self.encoder_filename = prefix
        self.directory_path = dir_path
        self.vocab_size = vocab_size
        self.out_name = '_encoded'
        self.imap = imap_type(dir_path, prefix, vocab_size) if imap_type else None

    @staticmethod
    def _get_files(path, filter_train=False, filter_words=None):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            # check whether the files are split into test, val set
            temp = get_files(path)
            temp = list(filter(lambda x: 'vocab.txt' not in x, temp))
            if filter_words:
                temp = list(filter(lambda x: filter_words in os.path.relpath(x, path), temp))
            # temp = list(filter(lambda x: '/raw' not in os.path.dirname(x), temp))
            trains = list(filter(lambda x: 'train' in os.path.basename(x), temp))
            if filter_train and trains:
                return trains
            else:
                return temp

    @abstractmethod
    def _read_file(self, file_path, **kwargs):
        # read file and returns line-wise tokenized strings
        # if a single line has multiple strings, such as source, target, out would be nested lists
        pass

    @abstractmethod
    def _encode_file(self, inp, out, **kwargs):
        # encode input file and save it at output filename
        pass

    @abstractmethod
    def _load_tokenizer(self, directory_path, encoder_filename):
        pass

    @abstractmethod
    def _learn_tokenizer(self, inp_path, **kwargs):
        pass

    @abstractmethod
    def _save_tokenizer(self, tokenizer):
        pass

    @abstractmethod
    def encode(self, texts):
        pass

    @abstractmethod
    def decode(self, indexed):
        pass

    @staticmethod
    def _save_df(df, out):
        if df is not None:
            if not os.path.exists(os.path.dirname(out)):
                os.makedirs(os.path.dirname(out))
            df.to_feather(out)

    def corpus_encode(self, inp_path, **kwargs):
        def output_path(path):
            if input_isdir:
                rel_path = os.path.relpath(path, inp_path)
                out = os.path.join(self.directory_path, inp_rel_path, 'encoded', rel_path)
                if os.path.splitext(out) != '.feather':
                    out += '.feather'
            else:
                out = os.path.join(self.directory_path, os.path.splitext(inp_path)[0] + '_encoded.pkl')  # deprecated
            return out

        if not self.is_trained:
            enc = self._learn_tokenizer(inp_path, **kwargs)
            self._save_tokenizer(enc)
            self.tokenizer, self.is_trained = self._load_tokenizer(self.directory_path, self.encoder_filename)

        procs = []

        input_isdir = os.path.isdir(inp_path)
        common_prefix = os.path.commonprefix([inp_path, self.directory_path])
        inp_rel_path = os.path.relpath(inp_path, common_prefix)
        fl = self._get_files(inp_path, **kwargs)
        outs = [output_path(i) for i in fl]

        for index, (inp, out) in enumerate(zip(fl, outs)):
            self._encode_file(inp, out, **kwargs)
            # proc = Process(target=self._encode_file, args=(inp, out))
        #     procs.append(proc)
        #     proc.start()
        # for proc in procs:
        #     proc.join()

        if self.imap is not None:
            self.imap.learn_dic(outs)
            self.imap.convert_corpus(outs)


class SpaceTokenizer(BaseTokenizer, ABC):
    def __init__(self, dir_path, prefix, vocab_size, use_sos=True, use_eos=True, **kwargs):
        super(SpaceTokenizer, self).__init__(dir_path, prefix, vocab_size, **kwargs)
        self.sos = use_sos
        self.eos = use_eos
        self.sos_token = '[SOS]'
        self.eos_token = '[EOS]'
        self.unk_token = 'UNK'

    def _learn_tokenizer(self, inp_path, **kwargs):
        full_path = os.path.join(self.directory_path, inp_path)
        fl = self._get_files(full_path, filter_train=True, **kwargs)
        cnt = collections.Counter()
        for i in fl:
            res = self._read_file(i)
            for j in res:
                j = j.split()
                if self.sos:
                    j = [self.sos_token] + j
                if self.eos:
                    j = j + [self.eos_token]
                cnt.update(j)
        if self.vocab_size:
            vocab_size = self.vocab_size
        else:
            vocab_size = len()
        mc = cnt.most_common(vocab_size - 2) + [('UNK', 1)]
        dic = dict(zip(map(lambda x: x[0], mc), range(len(mc))))
        return dic

    def _save_tokenizer(self, tokenizer):
        path = os.path.join(self.directory_path, self.encoder_filename + '.json')
        json.dump(tokenizer, open(path, 'w'))

    def _load_tokenizer(self, directory_path, encoder_filename):
        path = os.path.join(directory_path, encoder_filename + '.json')
        if os.path.exists(path):
            dic = load_json(path)
            inv_dic = dict(zip(dic.values(), dic.keys()))
            return (dic, inv_dic), True
        else:
            return None, False

    def encode(self, text):
        dic = self.tokenizer[0]
        text = text.split()
        if self.sos:
            text = [self.sos_token] + text
        if self.eos:
            text = text + [self.eos_token]
        encoded = [dic[i] if i in dic else len(dic) - 1 for i in text]
        return encoded

    def decode(self, indexed):
        inv_dic = self.tokenizer[1]
        res = [inv_dic[i] if i in inv_dic else 'PAD' for i in indexed]
        res = ' '.join(res)
        res = res.replace(self.eos_token, '')
        res = res.replace(self.sos_token, '')
        return res.strip()


class HFTokenizer(BaseTokenizer, ABC):  # Hugging Face tokenizers
    tokenizer_map = {'bpe':tokenizers.models.BPE, 'wp':tokenizers.models.WordPiece}
    trainer_map = {'bpe':tokenizers.trainers.BpeTrainer, 'wp':tokenizers.trainers.WordPieceTrainer}
    decoder_map = {'bpe':tokenizers.decoders.BPEDecoder, 'wp':tokenizers.decoders.WordPiece}
    default_special_tokens = ['[UNK]', '[SOS]', '[EOS]', '[MASK]']

    def __init__(self, dir_path, prefix, vocab_size=10000, tokenizer_class='wp',
                 morph_analyzer_class=MecabAnalyzer, cleanser_class=NullCleanser, tokens_to_add=None,
                 imap_type=IMap, split_jamo=False, **kwargs):
        if split_jamo:
            assert tokenizer_class.lower() == 'wp', \
                'Ja-mo level tokenization is only compatible with BertWordPieceTokenizer'
            self.space_symbol = 'ㅬ'
        else:
            self.space_symbol = 'ㅬ'
        self.morph_analyzer = morph_analyzer_class(space_symbol=self.space_symbol, jamo=split_jamo)
        self.cleanser = cleanser_class()
        self.imap = imap_type(dir_path, prefix, vocab_size, tokens_to_add) if imap_type else None
        self.split_jamo = split_jamo
        self.tokenizer_class = tokenizer_class.lower()
        self.tokens_to_add = tokens_to_add
        super(HFTokenizer, self).__init__(dir_path, prefix, vocab_size, imap_type, **kwargs)

    def token_to_id(self, token):
        id = self.tokenizer.token_to_id(token)
        if self.imap:
            id = self.imap.dic[str(id)]
        return id

    def _initialize_tokenizer(self):
        tokenizer = tokenizers.Tokenizer(self.tokenizer_map[self.tokenizer_class]())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace() # Todo: mecab support
        tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [SOS]:1",
            special_tokens=[
                ("[SOS]", self.default_special_tokens.index('[SOS]')),
                ("[EOS]", self.default_special_tokens.index('[EOS]')),
            ],
        )
        tokenizer.decoder = self.decoder_map[self.tokenizer_class]()
        return tokenizer

    def _load_tokenizer(self, directory_path, encoder_filename):
        def check_existence(inp):
            res = [os.path.exists(i) for i in inp]
            res = sum(res) == len(res)
            print(f'check {inp}\nresults : {res}')
            return res

        tokenizer = self._initialize_tokenizer()
        is_exists = False
        base_name = os.path.join(directory_path, encoder_filename)
        if self.tokenizer_class == 'wp':
            inp = (base_name + '-vocab.txt',)
        elif self.tokenizer_class == 'bpe':
            inp = (base_name + '-vocab.json', base_name + '-merges.txt')
        else:
            raise NotImplementedError
        if check_existence(inp):
            print('trained encoder loaded')
            is_exists = True
            tokenizer.model = self.tokenizer_map[self.tokenizer_class].from_file(*inp, unk_token='[UNK]')
        return tokenizer, is_exists

    def _write_one(self, inp, out, **kwargs):
        if os.path.exists(out):
            return
        tokenized_texts = self._read_file(inp, **kwargs)
        if isinstance(tokenized_texts[0], list):
            tokenized_texts = [' '.join(i) for i in tokenized_texts]
        with open(out, 'w', encoding='utf8') as f:
            tokenized_texts = [self.morph_analyzer.to_morphs(i, **kwargs) for i in tokenized_texts]
            f.writelines(tokenized_texts)

    def _write_to_txt(self, inp_path, out_path, **kwargs):
        input_isdir = os.path.isdir(inp_path)
        if not os.path.exists(out_path) and input_isdir:
            os.makedirs(out_path)
        fl = self._get_files(inp_path, filter_train=True, **kwargs)
        procs = []
        # for index, inp in enumerate(fl):
        #     basename = os.path.basename(inp)
        #     if input_isdir:
        #         out = os.path.join(out_path, basename + '.txt')
        #     else:
        #         out = out_path
        #     write_one(inp, out)
        filenames = []
        for index, inp in enumerate(fl):
            basename, _ = os.path.splitext(inp)
            basename = os.path.basename(basename)
            out = os.path.join(out_path, basename + '.txt')
            while out in filenames:
                out = os.path.join(out_path, basename + f'{random.randint(0,10000)}.txt')
            filenames.append(out)
            proc = Process(target=self._write_one, args=(inp, out))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

    def _learn_tokenizer(self, file_path, **kwargs):
        def merge_texts(out_path):
            base_name = os.path.dirname(out_path)
            fl = self._get_files(out_path, filter_train=True)
            with open(os.path.join(base_name, 'merged.txt'), 'w', encoding='utf8') as f:
                for i in fl:
                    with open(i, 'r', encoding='utf8') as t:
                        f.writelines(t.readlines())

        print('start encoder learning')
        if os.path.isdir(file_path):
            out_path = file_path + '_temp'
        else:
            out_path = os.path.splitext(file_path)[0] + '_temp.txt'
        self._write_to_txt(file_path, out_path, **kwargs)
        tokenizer = self.tokenizer
        merge_texts(out_path)
        base_name = os.path.dirname(out_path)
        self.istrained = True
        special_tokens = self.default_special_tokens
        if self.tokens_to_add:
            special_tokens += self.tokens_to_add
        trainer = self.trainer_map[self.tokenizer_class](vocab_size=self.vocab_size, special_tokens=special_tokens)
        tokenizer.train(trainer, [os.path.join(base_name, 'merged.txt')])
        print('finished encoder learning')
        #  remove cached files
        shutil.rmtree(out_path)
        os.remove(os.path.join(os.path.dirname(out_path),'merged.txt'))
        return tokenizer

    def _save_tokenizer(self, tokenizer):
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
        tokenizer.model.save(self.directory_path, self.encoder_filename)

    def encode(self, text, **kwargs):
        text = self.morph_analyzer.to_morphs(text, **kwargs)
        encoded = self.tokenizer.encode(text).ids
        if self.imap:
            encoded = self.imap.convert_line(encoded)
        return encoded

    def decode(self, indexed, **kwargs):
        if self.imap:
            indexed = self.imap.rollback_line(indexed)
        decoded = self.tokenizer.decode(indexed)
        decoded = self.morph_analyzer.to_texts(decoded, **kwargs)
        return decoded
