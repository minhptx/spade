import itertools
from functools import partial
from kbclean.datasets.dataset import Dataset
from kbclean.detection.features.base import BaseFeaturizer
from typing import Counter

import numpy as np
from kbclean.utils.data.helpers import str2regex
from kbclean.utils.data.attribute import (
    sym_trigrams,
    sym_value_freq,
    val_trigrams,
    value_freq,
    xngrams,
)
from loguru import logger


class HoloFeaturizer(BaseFeaturizer):
    def fit(self, values):
        trigram = [["".join(x) for x in list(xngrams(val, 3))] for val in values]
        ngrams = list(itertools.chain.from_iterable(trigram))
        self.trigram_counter = Counter(ngrams)
        sym_ngrams = [str2regex(x, False) for x in ngrams]

        self.sym_trigram_counter = Counter(sym_ngrams)
        self.val_counter = Counter(values)

        sym_values = [str2regex(x, False) for x in values]
        self.sym_val_counter = Counter(sym_values)

        self.func2counter = {
            val_trigrams: self.trigram_counter,
            sym_trigrams: self.sym_trigram_counter,
            value_freq: self.val_counter,
            sym_value_freq: self.sym_val_counter,
        }

    def transform(self, values):
        feature_lists = []
        for func, counter in self.func2counter.items():
            f = partial(func, counter=counter)
            logger.debug(
                "Negative: %s %s" % (func, list(zip(values[:10], f(values[:10]))))
            )
            logger.debug(
                "Positive: %s %s" % (func, list(zip(values[-10:], f(values[-10:]))))
            )
            feature_lists.append(f(values))

        feature_vecs = list(zip(*feature_lists))
        return np.asarray(feature_vecs)



