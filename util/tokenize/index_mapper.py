import pandas as pd
import collections
import itertools
from util.files import *
import argparse
import re


class IMap:
    def __init__(self, dir_path, prefix, vocab_size, added_special_tokens=None, **kwargs):
        self.dir_path = dir_path
        self.prefix = prefix
        self.probs_path, self.dic_path = self.get_path(dir_path, prefix)
        self.vocab_size = vocab_size
        self.added_special_tokens = added_special_tokens
        self.dic, self.inv_dic = self.load_dic(self.dic_path)
        self.target_names = kwargs.pop('target_names', None)

    @staticmethod
    def load_dic(path):
        if os.path.exists(path):
            dic = load_json(path)
            inv_dic = dict(zip(dic.values(), dic.keys()))
            return dic, inv_dic
        else:
            return None, None

    @staticmethod
    def get_path(dir_path, prefix):
        probs_path = os.path.join(dir_path, '{}-probs.json'.format(prefix))
        dic_path = os.path.join(dir_path, '{}-dic.json'.format(prefix))
        return probs_path, dic_path

    @staticmethod
    def get_columns(df_path):
        if isinstance(df_path,list):
            df_path = df_path[0]
        df = maybe_read(df_path)
        c = df.columns
        tar = set()
        for i in c:
            if isinstance(df[i][0], list) or isinstance(df[i][0], np.ndarray):
                tar.add(i)
        return tar

    @staticmethod
    def _get_files(path, filter_train=False):
        if os.path.isfile(path):
            res = os.path.join(os.path.splitext(path)[0] + '_encoded.pkl')
            return [res]
        elif os.path.isdir(path):
            if sum(['encoded' in i for i in os.listdir(path)]) == 0:  # if path is last directory
                path = path + '_encoded'
            # check whether the files are split into test, val set
            temp = get_files(path)
            temp = list(filter(lambda x: True in [os.path.basename(x).endswith('.pq'),
                                                  os.path.basename(x).endswith('.pkl'),
                                                  os.path.basename(x).endswith('.feather')], temp))
            trains = list(filter(lambda x: 'train' in os.path.basename(x), temp))
            if filter_train and trains:
                return trains
            else:
                return temp

    @staticmethod
    def new_filename(filename):
        nfn = re.sub(os.path.join('encoded', ''), os.path.join('encoded_mapped', ''), filename)
        return nfn

    def _count(self, path):
        vocab_size = self.vocab_size
        cnter = collections.Counter()
        s = set()
        checks = self.get_columns(path[0])
        if not self.target_names:
            self.target_names = checks
        targets = self.target_names
        for filename in path:
            cur_df = maybe_read(filename)
            for target in targets:
                texts = cur_df[target].tolist()
                for i in texts:
                    cnter.update(i[1:])
                    s.add(i[0])
            for check in checks:
                texts = cur_df[check].tolist()
                for i in texts:
                    s.update(i)

        # update missing vocabs
        sc = set(cnter)
        cnter.update(s.difference(sc))
        cnter.update(set(range(vocab_size)).difference(sc))

        # special tokens
        if self.added_special_tokens:
            cnter.update(self.added_special_tokens)

        tot = sum(cnter.values())
        cum_prob = [0]
        for i in cnter.most_common():
            cum_prob.append(cum_prob[-1] + i[1] / tot)
        cum_prob.pop(0)
        new_dict = dict([(int(old[0]), int(new)) for (new, old) in enumerate(cnter.most_common())])
        return cum_prob, new_dict

    def learn_dic(self, filepath):
        if not self.dic:
            print('start imap')
            probs, dic = self._count(filepath)
            inv_dic = dict(zip(dic.values(), dic.keys()))
            self.dic, self.inv_dic = dic, inv_dic
            json.dump(probs, open(self.probs_path, 'w'))
            json.dump(dic, open(self.dic_path, 'w'))
        else:
            print('imap exists')

    def convert_corpus(self, filepath):
        def _convert_file(filename, dic):
            key_type = type(list(dic)[0])
            cur_df = maybe_read(filename)
            for target in targets:
                new = []
                for line in cur_df[target].tolist():
                    converted = [dic[key_type(token)] if key_type(token) in dic else len(dic) - 1 for token in line]
                    new.append(converted)
                cur_df[target] = new
            return cur_df

        if not self.dic:
            self.learn_dic(filepath)
        if not self.target_names:
            self.target_names = self.get_columns(self._get_files(filepath))
        targets = self.target_names

        for filename in filepath:
            cur_df = _convert_file(filename, self.dic)
            new_filename = self.new_filename(filename)
            if not os.path.exists(os.path.dirname(new_filename)):
                os.makedirs(os.path.dirname(new_filename))
            cur_df.to_feather(new_filename)
        # else:
        #     cur_df = _convert_file(filepath[0], self.dic)
        #     new_path = os.path.splitext(filepath[0])[0] + '_mapped.pkl'
        #     cur_df.to_feather(new_path)

    def convert_line(self, line):
        assert self.dic is not None, 'dictionary must be built first'
        dic = self.dic
        for i in dic.keys():
            dic_key_type = type(i)
            break
        converted = []
        for token in line:
            converted.append(dic[dic_key_type(token)])
        return converted

    def rollback_line(self, indices):
        assert self.inv_dic is not None, 'dictionary must be built first'
        inv_dic = self.inv_dic
        new = []
        for ind in indices:
            if ind in inv_dic:
                new.append(int(inv_dic[ind]))
        return new


class MultilingualImap(IMap):
    def __init__(self, dir_path, prefix, vocab_size, added_special_tokens=None, **kwargs):
        super(MultilingualImap, self).__init__(dir_path, prefix, vocab_size, added_special_tokens, **kwargs)

    @staticmethod
    def extract_language(path):
        return os.path.basename(os.path.dirname(path))

    @staticmethod
    def get_path(dir_path, prefix):
        probs_path = os.path.join(dir_path, '{}-probs-ml.json'.format(prefix))
        dic_path = os.path.join(dir_path, '{}-dic-ml.json'.format(prefix))
        return probs_path, dic_path

    @staticmethod
    def new_filename(filename):
        sep = os.path.sep
        pattern = re.compile(f'encoded(_mapped)?{sep}')
        nfn = pattern.sub(f'encoded_mapped_ml{sep}', filename)
        return nfn

    @staticmethod
    def to_pops(tok, dics):
        rel_pos = [[i, dics[i][tok] / len(dics[i])] for i in dics if tok in dics[i]]
        to_pops = sorted(rel_pos, key=lambda x: x[-1])[1:]
        if to_pops:
            return list(map(lambda x:x[0], to_pops))
        else:
            return []

    @staticmethod
    def cat(lists):
        ml = min([len(i) for i in lists])
        cated = list(itertools.chain(*zip(*[i[:ml] for i in lists])))
        for i in lists:
            cated.extend(i[ml:])
        return cated

    def _count(self, path):
        vocab_size = self.vocab_size
        lang_cnters = {}
        s = set()
        checks = self.get_columns(path[0])
        if not self.target_names:
            self.target_names = checks
        targets = self.target_names
        for filename in path:
            lang = self.extract_language(filename)
            if lang not in lang_cnters:
                lang_cnters[lang] = collections.Counter()
            cur_df = maybe_read(filename)
            for target in targets:
                texts = cur_df[target].tolist()
                for i in texts:
                    lang_cnters[lang].update(i[1:])
                    s.add(i[0])
            for check in checks:
                texts = cur_df[check].tolist()
                for i in texts:
                    s.update(i)

        # update missing vocabs
        observed = set()
        [observed.update(i) for i in lang_cnters.values()]
        fl = list(lang_cnters.keys())[0]
        lang_cnters[fl].update(s.difference(observed))
        lang_cnters[fl].update(set(range(vocab_size)).difference(observed))

        # special tokens
        if self.added_special_tokens:
            lang_cnters[list(lang_cnters.keys())[0]] .update(self.added_special_tokens)
        dics_per_lang = {lang: {j[0]:i for i, j in enumerate(lang_cnters[lang].most_common())} for lang in lang_cnters}
        for tok in observed:
            tars = self.to_pops(tok, dics_per_lang)
            for tar in tars:
                dics_per_lang[tar].pop(tok)
        sorted_toks = [list(map(lambda x: x[0], sorted(dics_per_lang[i].items(), key=lambda x: x[-1])))
                             for i in dics_per_lang]
        cated = self.cat(sorted_toks)
        cum_prob = [0]  # deprecated
        new_dict = dict([(int(old), int(new)) for new, old in enumerate(cated)])
        return cum_prob, new_dict



def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base-name", type=str,
                        help='parent directory path')
    parser.add_argument("--dir-path", type=str,
                        help='directory where input data is stored')
    parser.add_argument("--target-filepath", type=str)
    parser.add_argument("--count-names", type=str, nargs='*')
    parser.add_argument("--check-names", type=str, nargs='*')
    parser.add_argument("--convert-names", type=str, nargs='*')
    return parser


if __name__ =='__main__':
    parser = get_parser()
    args = parser.parse_args()
    imap = MultilingualImap(args.dir_path, args.base_name, 60000)
    paths = get_files(args.target_filepath)
    imap.learn_dic(paths)
    imap.convert_corpus(paths)