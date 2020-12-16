from abc import ABCMeta, abstractmethod
from kbclean.datasets.dataset import Dataset
import numpy as np
import pandas as pd


class ErrorChecker(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, dirty_df: pd.DataFrame, col):
        pass

    def transform(self, dirty_df: pd.DataFrame, col):
        pass

    def fit_transform(self, dirty_df: pd.DataFrame, col, **kwargs):
        self.fit(dirty_df, col)

        return self.transform(dirty_df, col, **kwargs)


class BinaryErrorChecker(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, dirty_df: pd.DataFrame):
        pass

    def transform(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        pass

    def fit_transform(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        self.fit(dirty_df)

        return self.transform(dirty_df, lhs_col, rhs_col)

    def fit_transform_metal(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        scores = self.fit_transform(dirty_df, lhs_col, rhs_col)
        val_arr = np.ones((len(scores), 1)) * -1

        indices = np.where(scores == np.min(scores))[0]

        val_arr[indices] = 0.0

        return val_arr

