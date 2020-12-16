import itertools
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from kbclean.recommendation.checker.base import ErrorChecker
from kbclean.utils.data.attribute import xngrams
from kbclean.utils.inout import FastTextLoader
from spellchecker import SpellChecker
from torchtext.data.utils import get_tokenizer


@lru_cache()
def min_tok_ngram_counts(es_query, values):
    token_lists = list(
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
        self.model = SpellChecker()
        self.value2count = {}

    def fit(self, dirty_df: pd.DataFrame, col):
        col_values = dirty_df[col].values.tolist()

        self.value2count = dict(
            col_values,
            zip(
                list(
                    map(
                        lambda x: len(
                            self.model.unknown(
                                [tok for tok in nltk.wordpunct_tokenize(x)]
                            )
                        )
                        == 0,
                        col_values,
                    )
                )
            ),
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return np.asarray([self.value2count[value] for value in dirty_df[col].values])


class FastTextChecker(ErrorChecker):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")
        self.fasttext = FastTextLoader.get_instance()
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
        col_values = dirty_df[col].values.tolist()
        counts = [self._count_nonmeaning(str_value) for str_value in col_values]

        self.value2count = dict(zip(col_values, counts))

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray(
            [self.value2count[value] <= 0 for value in dirty_df[col].values.tolist()]
        )


class DictTypoTHChecker(ErrorChecker):
    def __init__(self):
        self.model = SpellChecker()
        self.value2count = {}

    def fit(self, dirty_df: pd.DataFrame, col):
        col_values = dirty_df[col].values.tolist()

        self.value2count = dict(
            col_values,
            zip(
                list(
                    map(
                        lambda x: len(
                            self.model.unknown(
                                [tok for tok in nltk.wordpunct_tokenize(x)]
                            )
                        )
                        == 0,
                        col_values,
                    )
                )
            ),
        )

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray(
            [self.value2count[value] > threshold for value in dirty_df[col].values]
        )


class WebTableChecker(ErrorChecker):
    def __init__(self, es_query):
        self.model = SpellChecker()
        self.es_query = es_query

    def fit(self, dirty_df: pd.DataFrame, col):
        col_values = dirty_df[col].values.tolist()
        counts = min_char_ngram_counts(self.es_query, tuple(col_values))

        self.value2count = dict(zip(col_values, counts))

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray(
            [self.value2count[value] > 1000 for value in dirty_df[col].values.tolist()]
        )


class FastTextTHChecker(ErrorChecker):
    def __init__(self):
        self.tokenizer = FastTextLoader.get_tokenizer()
        self.fasttext = FastTextLoader.get_instance()
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
        col_values = dirty_df[col].values.tolist()
        counts = [self._count_nonmeaning(str_value) for str_value in col_values]

        self.value2count = dict(zip(col_values, counts))

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray(
            [
                self.value2count[value] <= threshold
                for value in dirty_df[col].values.tolist()
            ]
        )


class WebTableTHChecker(ErrorChecker):
    def __init__(self, es_query):
        self.model = FastTextLoader.get_spell_checker()
        self.es_query = es_query

    def fit(self, dirty_df: pd.DataFrame, col):
        col_values = dirty_df[col].values.tolist()
        counts = min_char_ngram_counts(self.es_query, tuple(col_values))

        self.value2count = dict(zip(col_values, counts))

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray(
            [
                self.value2count[value] >= threshold
                for value in dirty_df[col].values.tolist()
            ]
        )

