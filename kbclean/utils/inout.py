import io

from argparse import Namespace
from pathlib import Path

import yaml


def load_config(config_path):
    config_path = Path(config_path)
    if config_path.is_dir():
        hparams = Namespace(**{})   
        for sub_path in config_path.iterdir():
            if sub_path.name.startswith("."):
                continue
            elif sub_path.stem == "config":
                hparams.__dict__.update(load_config(sub_path).__dict__)
            else:
                setattr(hparams, sub_path.stem, load_config(sub_path))
        return hparams
    else:
        param_dict = yaml.load(config_path.open("r"), Loader=yaml.FullLoader)
        hparams = Namespace(**param_dict)
        return hparams

def load_fasttext(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = map(float, tokens[1:])
    return data
