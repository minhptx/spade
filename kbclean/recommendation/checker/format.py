from collections import defaultdict
from typing import Counter
from kbclean.utils.data.helpers import str2regex
from kbclean.datasets.dataset import Dataset
from kbclean.recommendation.checker.base import ErrorChecker
import regex as re
import numpy as np
import pandas as pd
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import MLE
from nltk import word_tokenize


class CharChecker(ErrorChecker):
    def __init__(self):
        super().__init__()
        self.counter = None

    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        self.counter = Counter(values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.array([self.counter[value] for value in dirty_df[col].values])


class CharFormatChecker(CharChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        symbol_values = list(
            map(lambda x: str2regex(x, match_whole_token=False), values)
        )

        self.counter = Counter(symbol_values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):

        return np.array(
            [
                (
                    self.counter[str2regex(value, match_whole_token=False)]
                    * 1.0
                    / len(dirty_df)
                )
                for value in dirty_df[col].values
            ]
        )


class WordFormatChecker(CharChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        symbol_values = list(
            map(lambda x: str2regex(x, match_whole_token=True), values)
        )

        self.counter = Counter(symbol_values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):

        return np.asarray(
            [
                (
                    self.counter[str2regex(value, match_whole_token=True)]
                    * 1.0
                    / len(dirty_df)
                )
                for value in dirty_df[col].values
            ]
        )


class PunctFormatChecker(CharChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        symbol_values = list(map(lambda x: re.sub(r"[^?!,;.%$]+", "_", x), values))

        self.counter = Counter(symbol_values)
        print(self.counter)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray(
            [
                (self.counter[re.sub(r"[^?!,;.%$]+", "_", value)] * 1.0 / len(dirty_df))
                for value in dirty_df[col].values
            ]
        )


class CharTHChecker(ErrorChecker):
    def __init__(self):
        super().__init__()
        self.counter = None

    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        self.counter = Counter(values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.array(
            [self.counter[value] > threshold for value in dirty_df[col].values]
        )


class CharFormatTHChecker(CharTHChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        symbol_values = list(
            map(lambda x: str2regex(x, match_whole_token=False), values)
        )

        self.counter = Counter(symbol_values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        sum_count = sum(self.counter.values())

        return np.array(
            [
                (self.counter[str2regex(value, match_whole_token=False)] / sum_count)
                > threshold
                for value in dirty_df[col].values
            ]
        )


class WordFormatTHChecker(CharTHChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        symbol_values = list(
            map(lambda x: str2regex(x, match_whole_token=True), values)
        )

        self.counter = Counter(symbol_values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        sum_count = sum(self.counter.values())

        return np.asarray(
            [
                (self.counter[str2regex(value, match_whole_token=True)] / sum_count)
                > threshold
                for value in dirty_df[col].values
            ]
        )


class PunctFormatTHChecker(CharTHChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        values = dirty_df[col].values.tolist()
        symbol_values = list(map(lambda x: re.sub(r"[^?!,;.%$]+", "_", x), values))

        self.counter = Counter(symbol_values)

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        sum_count = sum(self.counter.values())

        return np.asarray(
            [
                (self.counter[re.sub(r"[^?!,;.%$]+", "_", value)] / sum_count)
                > threshold
                for value in dirty_df[col].values
            ]
        )


class PerplexityChecker(ErrorChecker):
    def __init__(self):
        self.model = MLE(2)

    def fit(self, dirty_df: pd.DataFrame, col):
        tokenized_text = [
            word_tokenize(value) for value in dirty_df[col].values
        ]

        train_data, padded_sents = padded_everygram_pipeline(2, tokenized_text)
        self.model.fit(train_data, padded_sents)

    def perplexity(self, sentence):
        tokenized_text = word_tokenize(sentence)
        tokenized_text = list(pad_both_ends(tokenized_text, 2))
        prob = 1
        for i in range(len(tokenized_text) - 1):
            prob = prob * self.model.score(tokenized_text[i + 1], [tokenized_text[i]])

        return (prob) ** (1.0/ len(tokenized_text))


    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray([self.perplexity(value) for value in dirty_df[col].values])




