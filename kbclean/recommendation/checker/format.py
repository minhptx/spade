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
import swifter


class CharChecker(ErrorChecker):
    def __init__(self):
        super().__init__()
        self.counter = None

    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = dirty_df[col].value_counts().to_dict()

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].swifter.apply(lambda x: self.counter[x] / len(dirty_df)).values


class CharFormatChecker(CharChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .swifter.apply(lambda x: str2regex(x, match_whole_token=False))
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .swifter.apply(
                lambda x: self.counter[str2regex(x, match_whole_token=False)]
                / len(dirty_df)
            )
            .values
        )


class WordFormatChecker(CharChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .swifter.apply(lambda x: str2regex(x, match_whole_token=True))
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .swifter.apply(
                lambda x: self.counter[str2regex(x, match_whole_token=True)]
                / len(dirty_df)
            )
            .values
        )


class PunctFormatChecker(CharChecker):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .swifter.apply(lambda x: re.sub(r"[^\p{P}]+", "_", x))
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .swifter.apply(
                lambda x: self.counter[re.sub(r"[^\p{P}]+", "_", x)] / len(dirty_df)
            )
            .values
        )


class PerplexityChecker(ErrorChecker):
    def __init__(self):
        self.model = MLE(2)

    def fit(self, dirty_df: pd.DataFrame, col):
        tokenized_text = [word_tokenize(value) for value in dirty_df[col].values]

        train_data, padded_sents = padded_everygram_pipeline(2, tokenized_text)
        self.model.fit(train_data, padded_sents)

    def perplexity(self, sentence):
        tokenized_text = word_tokenize(sentence)
        tokenized_text = list(pad_both_ends(tokenized_text, 2))
        prob = 1
        for i in range(len(tokenized_text) - 1):
            prob = prob * self.model.score(tokenized_text[i + 1], [tokenized_text[i]])

        return (prob) ** (1.0 / len(tokenized_text))

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].swifter.apply(lambda x: self.perplexity(x)).values

