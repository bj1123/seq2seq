import os
import yaml
from util.files import *
import argparse


class MTArgument:
    def __init__(self, path='config', is_train=True):
        training_path = os.path.join(path, 'training.yaml')
        model_data = os.path.join(path,'model.yaml')
        data = {}
        with open(training_path, "r") as t, open(model_data,'r') as m:
            training_data = yaml.load(t.read(), Loader=yaml.FullLoader)
            model_data = yaml.load(m.read(), Loader=yaml.FullLoader)
        self.is_train = is_train
        args = self.get_args(is_test=not is_train)
        args = args.parse_args()
        if args.model_size =='base':
            data.update(model_data['base'])
        else:
            data.update(model_data['large'])
        data.update(vars(args))
        data.update(training_data)
        self.load_files(data)
        self.__dict__ = data

    def get_args(self, is_test=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("--src-path", type=str, help='numpy arr path')
        parser.add_argument("--tgt-path", type=str, help='numpy arr path')
        parser.add_argument('--saved-model-folder', type=str)
        parser.add_argument('--saved-model-ckpt', type=str)
        parser.add_argument('--model-size', type=str, default='base')
        parser.add_argument("--loss-type", help="choice [plain, label-smoothing,"
                                                " losses that will be implemented in the future]",
                            type=str)
        parser.add_argument('--pre-lnorm', action='store_true')
        parser.add_argument("--model-checkpoint", help="transfer for finetune model",default="", type=str)
        return parser

    def load_files(self, data):
        dirname = os.path.join('data', 'saved_model')
        basename = '{}_{}'.format(data['model_size'], data['learning_rate'])
        data['train_src_path'] = os.path.join(data['src_path'],'train.pkl')
        data['train_tgt_path'] = os.path.join(data['tgt_path'],'train.pkl')
        data['test_src_path'] = os.path.join(data['src_path'],'test.pkl')
        data['test_tgt_path'] = os.path.join(data['tgt_path'],'test.pkl')
        data['padding_index'] = data['vocab_size'] - 1
        data['savename'] = os.path.join(dirname, basename)
        if data['saved_model_folder'] and data['saved_model_ckpt']:
            data['load_path'] = os.path.join(data['saved_model_folder'], data['saved_model_ckpt'])


class SamplingArgument(MTArgument):
    def __init__(self, path='config', is_train=True):
        super(SamplingArgument, self).__init__(path, is_train)

    def get_args(self, is_test=True):
        parser = super().get_args(is_test)
        parser.add_argument('--sample-save-path', type=str)
        parser.add_argument('--sampling-mode', type=str)
        parser.add_argument('--width', type=int)
        parser.add_argument('--temperature', type=float, default=1.0)
        return parser