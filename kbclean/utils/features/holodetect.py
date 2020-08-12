import itertools
from collections import Counter

import numpy as np
import regex as re
from nltk import ngrams, wordpunct_tokenize
from regex.regex import match

from kbclean.utils.data.helpers import str2regex


def xngrams(value, n):
    value = "^" + value + "$"
    try:
        return list(set(ngrams(value, n)))
    except:
        return ["^$"]


def val_trigrams(values):
    val_trigrams = [["".join(x) for x in list(xngrams(val, 3))] for val in values]
    ngrams = itertools.chain.from_iterable(val_trigrams)
    counter = Counter(ngrams)
    sum_count = sum(counter.values())

    res = [
        (min([counter[gram] for gram in trigram]) / sum_count if trigram else 0) * 1.0
        for trigram in val_trigrams
    ]
    return res


def sym_trigrams(values):
    patterns = list(map(lambda x: str2regex(x, False), values))
    return val_trigrams(patterns)


def value_freq(values):
    counter = Counter(values)
    sum_couter = sum(counter.values())
    return [counter[value] * 1.0 / sum_couter for value in values]


def sym_value_freq(values):
    patterns = list(map(lambda x: str2regex(x, True), values))

    return value_freq(patterns)


def has_x(values):
    return ["%" in val for val in values]
