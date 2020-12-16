from typing import Counter
import numpy as np
import torch
from kbclean.datasets.dataset import Dataset
from kbclean.detection.features.base import BaseFeaturizer
from kbclean.utils.data.helpers import str2regex
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from ftfy import fix_text


class StatsFeaturizer(BaseFeaturizer):
    def __init__(self):
        self.char_counter = CountVectorizer(
            tokenizer=lambda x: list(x), lowercase=False, binary=True
        )
        self.regex_counter = CountVectorizer(
            tokenizer=lambda x: list(x), lowercase=False, binary=True
        )
        self.regex_counter2 = CountVectorizer(
            tokenizer=lambda x: list(x), lowercase=False, binary=True
        )

        self.covalue_counter = {}
        self.value_counter = {}

    def fit(self, dirty_df: pd.DataFrame, col: str):
        self.char_counter.fit(dirty_df[col].values.tolist())
        self.regex_counter.fit(
            [str2regex(val, match_whole_token=False) for val in dirty_df[col].values]
        )
        self.regex_counter2.fit(
            [str2regex(val, match_whole_token=True) for val in dirty_df[col].values]
        )

        self.value_counter = dict(Counter(dirty_df[col].values))

    def transform(self, dirty_df: pd.DataFrame, col: str):
        char_features = self.char_counter.transform(
            dirty_df[col].values.tolist()
        ).todense()
        # word_features = self.word_counter.transform(
        #     dataset.dirty_df[col].values.tolist()
        # ).todense()
        regex_features = self.regex_counter.transform(
            [str2regex(val, match_whole_token=False) for val in dirty_df[col].values]
        ).todense()

        regex_features2 = self.regex_counter2.transform(
            [str2regex(val, match_whole_token=False) for val in dirty_df[col].values]
        ).todense()

        return [
            torch.tensor(
                np.concatenate([char_features, regex_features, regex_features2], axis=1)
            )
        ]

    def n_features(self, dirty_df: pd.DataFrame):
        return sum(
            [
                len(x.get_feature_names())
                for x in [self.char_counter, self.regex_counter, self.regex_counter2]
            ]
        )

    def feature_dim(self):
        return 1

