import io

from argparse import Namespace
from pathlib import Path
from spellchecker.spellchecker import SpellChecker
from torchtext.vocab import FastText
from torchtext.data.utils import get_tokenizer


import yaml

class FastTextLoader:
    instance = None
    tokenizer = None
    spell_checker = None

    @staticmethod
    def get_instance():
        if FastTextLoader.instance is None:
            FastTextLoader.instance = FastText()
        return FastTextLoader.instance

    @staticmethod
    def get_tokenizer():
        if FastTextLoader.tokenizer is None:
            FastTextLoader.tokenizer = get_tokenizer("spacy")
        return FastTextLoader.tokenizer

    @staticmethod
    def get_spell_checker():
        if FastTextLoader.spell_checker is None:
            FastTextLoader.spell_checker = SpellChecker()
        return FastTextLoader.spell_checker



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
