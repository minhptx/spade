import itertools
from functools import lru_cache
from kbclean.utils.search.query import ESQuery

import nltk
import numpy as np
import pandas as pd
import swifter
from kbclean.recommendation.checker.base import ErrorChecker
from kbclean.utils.data.attribute import xngrams
from kbclean.utils.inout import SingletonLoader
from spellchecker import SpellChecker
from torchtext.data.utils import get_tokenizer


@lru_cache()
def min_tok_ngram_counts(es_query, values):
    token_lists = set(
        itertools.chain.from_iterable(
            [nltk.wordpunct_tokenize(value) for value in values]
        )
    )

    counts = es_query.get_tok_ngram_counts(token_lists)

    value2count = {trigram: int(count) for trigram, count in zip(token_lists, counts)}

    def get_min_count(value):
        if not value:
            return 0
        tokens = [x for x in nltk.wordpunct_tokenize(value) if x.isalpha()]
        if tokens:
            return min([value2count[x] for x in tokens])
        else:
            return 100000

    return [get_min_count(value) for value in values]


@lru_cache()
def min_char_ngram_counts(es_query, values):
    bigrams = list(
        itertools.chain.from_iterable([xngrams(value, 2, False) for value in values])
    )

    counts = es_query.get_char_ngram_counts(bigrams)

    value2count = {trigram: int(count) for trigram, count in zip(bigrams, counts)}

    def get_min_count(value):
        if not value:
            return 0
        tokens = [x for x in xngrams(value, 2, False)]
        if tokens:
            return min([value2count[x] for x in tokens])
        else:
            return 100000

    return [get_min_count(value) for value in values]


class DictTypoChecker(ErrorChecker):
    def __init__(self):
        self.model = SingletonLoader.get_spell_checker()
        self.value2count = {}

    def fit(self, dirty_df: pd.DataFrame, col):
        self.value2count = dict(
            dirty_df[col]
            .swifter.apply(
                lambda x: (
                    x,
                    len(self.model.unknown([tok for tok in nltk.wordpunct_tokenize(x)]))
                    == 0,
                )
            )
            .values
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].swifter.apply(lambda x: self.value2count[x]).values


class FastTextChecker(ErrorChecker):
    def __init__(self):
        self.tokenizer = SingletonLoader.get_tokenizer()
        self.fasttext = SingletonLoader.get_instance()
        self.value2count = {}

    @lru_cache()
    def _count_nonmeaning(self, str_value):
        try:
            result = np.count_nonzero(
                [
                    (x.sum() == 0)
                    for x in self.fasttext.get_vecs_by_tokens(
                        self.tokenizer(str_value), lower_case_backup=True
                    )
                ]
            )

            return result
        except Exception as e:
            print(e)
            return 0

    def fit(self, dirty_df: pd.DataFrame, col):
        self.value2count = dict(
            dirty_df[col]
            .swifter.apply(lambda x: (x, self._count_nonmeaning(x) == 0))
            .values
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].swifter.apply(lambda x: self.value2count[x]).values


class WebTableBoolChecker(ErrorChecker):
    def __init__(self, bigram_path):
        self.bigram_dict = SingletonLoader.get_bigram_dict(bigram_path)

    def fit(self, dirty_df: pd.DataFrame, col):
        pass

    def transform(self, dirty_df: pd.DataFrame, col):
        def get_min_count(value):
            if not value:
                return 0
            tokens = [x for x in xngrams(value, 2, False)]
            if tokens:
                return min([self.bigram_dict.get(x, 0) for x in tokens])
            else:
                return 100000
        return (
            dirty_df[col]
            .swifter.apply(
                lambda x: get_min_count(x) / 10000
            ).apply(lambda x: x if x < 1 else 1)
            .values
        )


class WebTableChecker(ErrorChecker):
    def __init__(self, es_query):
        self.es_query = es_query

    def fit(self, dirty_df: pd.DataFrame, col):
        col_values = dirty_df[col].values.tolist()
        counts = min_char_ngram_counts(self.es_query, tuple(col_values))

        self.value2count = dict(zip(col_values, counts))

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].swifter.apply(lambda x: self.value2count[x]).values
