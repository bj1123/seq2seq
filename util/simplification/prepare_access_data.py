from access.preprocessors import get_preprocessors
from access.resources.datasets import create_preprocessed_dataset
import shutil
from access.resources.paths import get_dataset_dir
from access.access.fairseq.main import prepare_exp_dir

def main():
    preprocessors = get_preprocessors(preprocessors_kwargs)
    dataset = 'wikilarge'
    exp_dir = prepare_exp_dir()
    dataset = create_preprocessed_dataset(dataset, preprocessors, n_jobs=1)
    shutil.copy(get_dataset_dir(dataset) / 'preprocessors.pickle', exp_dir)
    return dataset

if __name__ =='__main__':
    main()