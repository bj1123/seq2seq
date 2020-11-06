import os
import yaml
from util.files import *
import argparse


class LMArgument:
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
        parser.add_argument("--filpath", type=str, default='temp.npy',
                            help='numpy arr path')
        parser.add_argument('--saved-path', type=str)
        parser.add_argument('--model-size', type=str, default='base')
        parser.add_argument("--loss-type", help="choice [plain, losses that will be implemented in the future]",
                            required=True, type=str)
        parser.add_argument('--pre-lnorm', action='store_true')
        parser.add_argument("--model-checkpoint", help="transfer for finetune model",default="", type=str)
        return parser

    def load_files(self, data):
        dirname = os.path.join('data','saved_model')
        basename = '{}_{}'.format(data['model_size'], data['learning_rate'])

        data['vocab_size'] = 6
        data['padding_index'] = data['vocab_size']
        data['savename'] = os.path.join(dirname, basename)


class SamplingArgument(LMArgument):
    def __init__(self, path='config', is_train=True):
        super(SamplingArgument, self).__init__(path, is_train)

    def load_files(self, data):
        super().load_files(data)
        data['encoder_dir'] = os.path.join(data['root'],data['dataset'])

    def get_args(self, is_test=True):
        parser = super().get_args(is_test)
        parser.add_argument('--decoded-path', type=str)
        parser.add_argument('--encoder-name', type=str)
        return parser