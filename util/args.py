import os
import yaml
from util.files import *
import argparse
from abc import ABC, abstractmethod


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-sampling", action='store_true')
    parser.add_argument("--task", type=str)
    parents_args, _ = parser.parse_known_args()
    if parents_args.task == 'seq2seq':
        args = MTArgument(parser, is_train=not parents_args.is_sampling)
    elif parents_args.task == 'mnmt':
        args = MNMTArgument(parser, is_train=not parents_args.is_sampling)
    elif parents_args.task == 'multitask':
        args = MultitaskArgument(parser, is_train=not parents_args.is_sampling)
    elif parents_args.task == 'access':
        args = AccessArgument(parser, is_train=not parents_args.is_sampling)
    else:
        raise NotImplementedError
    return args


class BaseArgument(ABC):
    def __init__(self, parents_args, path='config', is_train=True):
        self.parents_args = parents_args
        training_path = os.path.join(path, 'training.yaml')
        model_data = os.path.join(path, 'model.yaml')
        data = {}
        with open(training_path, "r") as t, open(model_data, 'r') as m:
            training_data = yaml.load(t.read(), Loader=yaml.FullLoader)
            model_data = yaml.load(m.read(), Loader=yaml.FullLoader)
        self.is_train = is_train
        args = self.get_args(is_test=not is_train)
        args = args.parse_args()
        if args.model_size == 'base':
            data.update(model_data['base'])
        else:
            data.update(model_data['large'])
        data.update(vars(args))
        data.update(training_data)
        self.load_files(data)
        self.__dict__ = data

    def get_common_args(self, is_test=False):
        parser = self.parents_args
        parser.add_argument('--saved-model-folder', type=str)
        parser.add_argument('--saved-model-ckpt', type=str)
        parser.add_argument('--model-size', type=str, default='base')
        parser.add_argument('--model-type', type=str, default='plain')
        parser.add_argument("--loss-type", help="choice [plain, label-smoothing,"
                                                " losses that will be implemented in the future]",
                            type=str)
        parser.add_argument('--positional-encoding', type=str, default='absolute')
        if is_test:
            parser.add_argument('--sampling-mode', type=str)
            parser.add_argument('--width', type=int)
            parser.add_argument('--temperature', type=float, default=1.0)
            parser.add_argument('--lengths-penalty', type=float, default=1.0)

        return parser

    @abstractmethod
    def load_files(self, data):
        pass

    @abstractmethod
    def get_args(self, is_test=False):
        pass


class MTArgument(BaseArgument):
    def __init__(self, parents_args, path='config', is_train=True):
        super(MTArgument, self).__init__(parents_args, path, is_train)

    @staticmethod
    def get_indices(cum_prob, target_probs=[0.4, 0.7, 0.9]):
        cur = 0
        res = []
        for i in target_probs:
            while cum_prob[cur] < i:
                cur += 1
            res.append(cur)
        return res

    def get_args(self, is_test=False):
        parser = self.get_common_args(is_test)
        parser.add_argument("--src-path", type=str)
        parser.add_argument("--tgt-path", type=str)
        parser.add_argument("--dataset-name", type=str)
        parser.add_argument('--prob-path', type=str)
        parser.add_argument("--model-checkpoint", help="transfer for finetune model", default="", type=str)

        return parser

    def load_files(self, data):
        model_type = data['model_type']
        dirname = os.path.join('data', 'saved_model', data['dataset_name'], model_type)
        basename = '{}_{}_{}'.format(data['model_size'], data['learning_rate'], data['positional_encoding'])
        # for debugging
        data['train_src_path'] = files_including(data['src_path'], 'train')
        data['train_tgt_path'] = files_including(data['tgt_path'], 'train')
        data['train_example_path'] = os.path.join(os.path.dirname(data['tgt_path']), 'tr_examples')
        data['test_src_path'] = files_including(data['src_path'], 'test')
        data['test_tgt_path'] = files_including(data['tgt_path'], 'test')
        data['test_example_path'] = os.path.join(os.path.dirname(data['tgt_path']), 'te_examples')
        data['src_padding_index'] = data['src_vocab_size'] - 1
        data['tgt_padding_index'] = data['tgt_vocab_size'] - 1
        if data['prob_path']:
            data['cum_probs'] = load_json(data['prob_path'])
            data['cutoffs'] = self.get_indices(data['cum_probs'])
            data['rare_index'] = data['cutoffs'][-1]
        data['savename'] = os.path.join(dirname, basename)
        if data['saved_model_folder'] and data['saved_model_ckpt']:
            data['load_path'] = os.path.join(data['saved_model_folder'], data['saved_model_ckpt'])
            data['sample_save_path'] = data['load_path'].replace('saved_model', 'sampled')
            if f'{os.path.sep}semi{os.path.sep}' in data['saved_model_folder']:
                data['sample_save_path'] = data['sample_save_path'] +\
                                           '_' +os.path.basename(data['src_path'])


class MultitaskArgument(BaseArgument):
    def __init__(self, parents_args, path='config', is_train=True):
        super(MultitaskArgument, self).__init__(parents_args, path, is_train)

    @staticmethod
    def _get_filepath(path):
        from util.tokenize.data_reformatter import MultitaskReformatter
        files = files_including(path, '_encoded_mapped')
        base_dirs = [os.path.basename(os.path.dirname(i)).replace('_encoded_mapped', '') for i in files]
        tm = MultitaskReformatter.tasks_map
        tasks = [tm[i] if i in tm else None for i in base_dirs]
        task_set = set(tasks)
        file_paths = {task: [] for task in task_set}
        for ind, task in enumerate(tasks):
            file_paths[task].append(files[ind])
        if None in file_paths:
            file_paths.pop(None)
        return file_paths

    @staticmethod
    def _get_special_tokens_indice(data):
        from util.tokenize.data_specific_tokenizer import MultiTaskTokenizer
        from util.tokenize.data_reformatter import MultitaskReformatter
        tokenizer = MultiTaskTokenizer(data['dir_path'], data['tokenizer_prefix'],
                                       tokenizer_class=data['tokenizer_class'])
        tasks = list(MultitaskReformatter.tasks_map.values())
        task_dic = {i: tokenizer.token_to_id(i) for i in tasks}
        languages = MultitaskReformatter.languages
        language_dic = {i: tokenizer.token_to_id(i) for i in languages}
        symbols = MultiTaskTokenizer.default_special_tokens
        symbol_dic = {i: tokenizer.token_to_id(i) for i in symbols}

        indice = {'language': language_dic, 'task': task_dic, 'symbols': symbol_dic}
        return indice

    def get_args(self, is_test=False):
        parser = self.get_common_args()
        parser.add_argument("--dir-path", type=str)
        parser.add_argument("--tokenizer-prefix", type=str)
        parser.add_argument("--tokenizer-class", type=str, default='wp')
        parser.add_argument("--model-checkpoint", help="transfer for finetune model", default="", type=str)
        if is_test:
            parser.add_argument('--target-text-path', type=str)
            parser.add_argument('--sampling-mode', type=str)
            parser.add_argument('--width', type=int)
            parser.add_argument('--temperature', type=float, default=1.0)
            parser.add_argument('--lengths-penalty', type=float, default=1.0)
            return parser
        return parser

    def load_files(self, data):
        dirname = os.path.join('data', 'saved_model', 'multitask')
        basename = '{}_{}'.format(data['model_size'], data['learning_rate'])
        file_paths = self._get_filepath(data['dir_path'])
        data['special_token_indice'] = self._get_special_tokens_indice(data)
        data['train_path'] = {i: list(filter(lambda k: 'train' in os.path.basename(k), file_paths[i]))
                              for i in file_paths.keys()}
        data['test_path'] = {i: list(filter(lambda k: 'test' in os.path.basename(k), file_paths[i]))
                              for i in file_paths.keys()}
        data['padding_index'] = data['vocab_size'] - 1
        data['savename'] = os.path.join(dirname, basename)
        if data['saved_model_folder'] and data['saved_model_ckpt']:
            data['load_path'] = os.path.join(data['saved_model_folder'], data['saved_model_ckpt'])


class MNMTArgument(BaseArgument):
    def __init__(self, parents_args, path='config', is_train=True):
        super(MNMTArgument, self).__init__(parents_args, path, is_train)

    def get_args(self, is_test=False):
        parser = self.get_common_args(is_test)
        parser.add_argument("--dir-path", type=str)
        parser.add_argument("--dataset-name", type=str)
        parser.add_argument("--target-lang", type=str)
        parser.add_argument("--n_chunk", type=int, default=1)
        if is_test:
            parser.add_argument('--sample-save-path', type=str)
        return parser

    @staticmethod
    def pair_files(dir_path, target_name='train', target_lang=None):
        dirs = os.listdir(dir_path)
        res = []
        for i in dirs:
            files = files_including(os.path.join(dir_path, i), target_name)
            if target_lang:
                files = list(filter(lambda x: target_lang in os.path.relpath(x, dir_path), files))
                if files and target_lang in os.path.basename(os.path.dirname(files[0])):
                    files[1], files[0] = files[0], files[1]
            if files:
                res.append(files)
        return res

    def load_files(self, data):
        model_type = data['model_type']
        middle_folder = os.path.join('semi', data['target_lang']) if data['target_lang'] else 'multi'
        dirname = os.path.join('data', 'saved_model', data['dataset_name'], middle_folder, model_type)
        basename = '{}_{}_{}'.format(data['model_size'], data['learning_rate'], data['positional_encoding'])
        # for debugging
        target_lang = f'_{data["target_lang"]}' if data['target_lang'] else ''
        mode = '' if model_type == 'plain' else f'_{model_type}'
        data['train_path'] = self.pair_files(data['dir_path'], 'train', data['target_lang'])
        data['train_example_path'] = os.path.join(data['dir_path'], 'tr_examples' + target_lang + mode)
        data['test_path'] = self.pair_files(data['dir_path'], 'test', data['target_lang'])
        data['test_example_path'] = os.path.join(data['dir_path'], 'te_examples' + target_lang + mode)
        data['padding_index'] = data['vocab_size'] - 1
        data['savename'] = os.path.join(dirname, basename)
        if data['saved_model_folder'] and data['saved_model_ckpt']:
            data['load_path'] = os.path.join(data['saved_model_folder'], data['saved_model_ckpt'])


class AccessArgument(BaseArgument):
    def __init__(self, parents_args, path='config', is_train=True):
        super(AccessArgument, self).__init__(parents_args, path, is_train)

    def get_args(self, is_test=False):
        parser = self.get_common_args()
        parser.add_argument("--loss-type", help="choice [plain]"
                                                " losses that will be implemented in the future]",
                            type=str)

        parser.add_argument("--dataset", type=str)
        if is_test:
            parser.add_argument('--sample-save-path', type=str)
            parser.add_argument('--sampling-mode', type=str)
            parser.add_argument('--width', type=int)
            parser.add_argument('--temperature', type=float, default=1.0)
            parser.add_argument('--lengths-penalty', type=float, default=1.0)
        return parser

    def load_files(self, data):
        model_type = data['model_type']
        dirname = os.path.join('data', 'saved_model', 'access', model_type)
        basename = '{}_{}'.format(data['model_size'], data['learning_rate'])
        data['savename'] = os.path.join(dirname, basename)
        data['padding_index'] = 1
        if data['saved_model_folder'] and data['saved_model_ckpt']:
            data['load_path'] = os.path.join(data['saved_model_folder'], data['saved_model_ckpt'])
