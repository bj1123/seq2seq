import os
import glob
import json
import torch
import pandas as pd


def get_files(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)
    return paths


def file_iterator(path):
    files = get_files(path)
    for file in files:
        f = open(file, errors='ignore')
        yield f.read()


def check_file(filename):
    return os.path.exists(filename)


def load_json(filename):
    with open(filename, 'r') as f:
        v = json.load(f)
    return v


def ckpt_average(ckptlists):
    base = torch.load(ckptlists[0])
    for ckpt_filename in ckptlists[1:]:
        ckpt = torch.load(ckpt_filename)
        for i in ckpt:
            base[i] += ckpt[i]
    for i in base:
        base[i] /= len(ckptlists)
    return base


def files_including(path, str_to_include):
    fl = get_files(path)
    return list(filter(lambda x: str_to_include in x, fl))


def df_to_txt(inp_path, out_path):
    df = pd.read_pickle(inp_path)
    with open(out_path, 'w') as f:
        res = [' '.join(map(str, i[1:-1])) + ' \n' for i in df.texts]
        f.writelines(res)




