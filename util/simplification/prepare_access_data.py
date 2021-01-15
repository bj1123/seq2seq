from access.preprocessors import get_preprocessors
from access.resources.datasets import create_preprocessed_dataset
import shutil
from access.resources.paths import get_dataset_dir
from access.fairseq.main import prepare_exp_dir
from access.resources.prepare import prepare_wikilarge, prepare_turkcorpus


def main():
    prepare_wikilarge()
    prepare_turkcorpus()
    preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.8},
        'LevenshteinPreprocessor': {'target_ratio': 0.8},
        'WordRankRatioPreprocessor': {'target_ratio': 0.8},

        'DependencyTreeDepthRatioPreprocessor': {
            'target_ratio': 0.8  # Default initial value
        },
        'SentencePiecePreprocessor': {
            'vocab_size': 10000
        }
    }
    preprocessors = get_preprocessors(preprocessors_kwargs)
    dataset = 'wikilarge'
    exp_dir = prepare_exp_dir()
    dataset = create_preprocessed_dataset(dataset, preprocessors, n_jobs=1)
    shutil.copy(get_dataset_dir(dataset) / 'preprocessors.pickle', exp_dir)
    print(dataset)
    return dataset

if __name__ =='__main__':
    main()