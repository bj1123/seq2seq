import pandas as pd
import collections
from util.files import *
import argparse


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
        df = pd.read_pickle(df_path)
        c = df.columns
        tar = set()
        for i in c:
            if isinstance(df[i][0], list):
                tar.add(i)
        return tar

    @staticmethod
    def _get_files(path, filter_train=False):
        if os.path.isfile(path):
            res = os.path.join(os.path.splitext(path)[0] + '_encoded.pkl')
            return [res]
        elif os.path.isdir(path):
            if sum([i.endswith('.pkl') for i in os.listdir(path)]) >= len(os.listdir(path))-1:  # if path is last directory
                path = path + '_encoded'
            # check whether the files are split into test, val set
            temp = get_files(path)
            temp = list(filter(lambda x: os.path.dirname(x).endswith('encoded'), temp))
            trains = list(filter(lambda x: 'train' in os.path.basename(x), temp))
            if filter_train and trains:
                return trains
            else:
                return temp

    def _count(self, path):
        vocab_size = self.vocab_size
        cnter = collections.Counter()
        s = set()
        fl = self._get_files(path, filter_train=True)
        checks = self.get_columns(fl[0])
        if not self.target_names:
            self.target_names = checks
        targets = self.target_names
        for filename in fl:
            cur_df = pd.read_pickle(filename)
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
            cur_df = pd.read_pickle(filename)
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

        fl = self._get_files(filepath)
        if os.path.isdir(filepath):
            for filename in fl:
                base_filename = os.path.basename(filename)
                dirname = os.path.dirname(filename)
                cur_df = _convert_file(filename, self.dic)
                new_filename = os.path.join(dirname + '_mapped', base_filename)
                if not os.path.exists(os.path.dirname(new_filename)):
                    os.makedirs(os.path.dirname(new_filename))
                cur_df.to_pickle(new_filename)
        else:
            cur_df = _convert_file(fl[0], self.dic)
            new_path = os.path.splitext(fl[0])[0] + '_mapped.pkl'
            cur_df.to_pickle(new_path)

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


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base-name", type=str,
                        help='parent directory path')
    parser.add_argument("--dir-path", type=str,
                        help='directory where input data is stored')
    parser.add_argument("--count-names", type=str, nargs='*')
    parser.add_argument("--check-names", type=str, nargs='*')
    parser.add_argument("--convert-names", type=str, nargs='*')
    return parser


if __name__ =='__main__':
    parser = get_parser()
    args = parser.parse_args()
    imap = IMap(args.dir_path, args.base_name)
    imap.learn_dic(args.count_names, args.check_names)
    imap.convert_and_save(args.convert_names)