from kbclean.datasets.dataset import Dataset
from kbclean.recommendation.checker.base import ErrorChecker
import numpy as np
import pandas as pd

class MissingValueChecker(ErrorChecker):
    def __init__(self):
        pass

    def fit(self, dirty_df: pd.DataFrame, col):
        pass

    def transform(self, dirty_df: pd.DataFrame, col, threshold):
        return np.asarray([0 if val.strip().lower() in ["", "null", "none", "na"] else 1 for val in dirty_df[col].values])