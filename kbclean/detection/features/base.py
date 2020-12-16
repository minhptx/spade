import abc
from abc import abstractmethod
import itertools

import torch
from typing import List
import numpy as np
import pandas as pd


class BaseFeaturizer(metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(self, dirty_df: pd.DataFrame, col: str):
        pass

    @abstractmethod
    def transform(self, dirty_df: pd.DataFrame, col: str):
        pass

    def fit_transform(self, dirty_df: pd.DataFrame, col: str):
        self.fit(dirty_df, col)

        return self.transform(dirty_df, col)

    @abstractmethod
    def n_features(self, dirty_df: pd.DataFrame):
        pass

    @abstractmethod
    def feature_dim(self):
        return 1


class UnionFeaturizer(BaseFeaturizer):
    def __init__(self, name2extractor):
        self.name2extractor = name2extractor

    def fit(self, dirty_df: pd.DataFrame, col: str):
        for extractor in self.name2extractor.values():
            extractor.fit(dirty_df, col)

    def transform(self, dirty_df: pd.DataFrame, col: str):
        return list(
            itertools.chain.from_iterable(
                [
                    extractor.transform(dirty_df, col)
                    for extractor in self.name2extractor.values()
                ]
            )
        )

    def __getitem__(self, key):
        return self.name2extractor[key]

    def n_features(self, dirty_df: pd.DataFrame):
        return sum([x.n_features(dirty_df) for x in self.name2extractor.values()])

    def feature_dim(self):
        return list(self.name2extractor.values())[0].feature_dim()


class ConcatFeaturizer(UnionFeaturizer):
    def __init__(self, name2extractor):
        self.name2extractor = name2extractor

    def transform(self, dirty_df: pd.DataFrame, col: str):
        return torch.cat(
            list(
                itertools.chain.from_iterable(
                    [
                        extractor.transform(dirty_df, col)
                        for extractor in self.name2extractor.values()
                    ]
                )
            ),
            axis=1,
        )

    def __getitem__(self, key):
        return self.name2extractor[key]

    def n_features(self, dirty_df: pd.DataFrame):
        return sum([x.n_features(dirty_df) for x in self.name2extractor.values()])

    def feature_dim(self):
        return list(self.name2extractor.values())[0].feature_dim()
