import itertools
from collections import Counter

import numpy as np
import regex as re
from nltk import trigrams, wordpunct_tokenize



def str2regex(x):
    if x is None:
        return ""
    x = re.sub(r"[A-Z]", "A", x)
    x = re.sub(r"[a-z]", "a", x)
    x = re.sub(r"[0-9]", "0", x)
    return x


def char_fasttext(values, model):
    def get_feature(v):
        return np.mean([model[c] for c in v])

    return [get_feature(val) for val in values]


def word_fasttext(values, model):
    def get_feature(v):
        return np.mean([model[w] for w in wordpunct_tokenize(v)])

    return [get_feature(val) for val in values]


def val_trigrams(values):
    val_trigrams = [[''.join(x) for x in list(trigrams(val))] for val in values]
    ngrams = itertools.chain.from_iterable(val_trigrams)
    counter = Counter(ngrams)
    sum_count = sum(counter.values())

    res = [
        min([counter[gram] for gram in trigram]) / sum_count if trigram else 0 * 1.0
        for trigram in val_trigrams
    ]
    return res


def sym_trigrams(values):
    patterns = list(map(str2regex, values))
    return val_trigrams(patterns)


def value_freq(values):
    counter = Counter(values)
    sum_couter = sum(counter.values())
    return [counter[value] * 1.0 / sum_couter for value in values]


def sym_value_freq(values):
    patterns = list(map(str2regex, values))

    return value_freq(patterns)


def has_x(values):
    return ["%" in val for val in values]