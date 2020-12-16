from collections import defaultdict
from typing import Counter

from kbclean.datasets.dataset import Dataset
from kbclean.recommendation.checker.base import BinaryErrorChecker, ErrorChecker
import numpy as np
import pandas as pd

class FDChecker(BinaryErrorChecker):
    def fit(self, dirty_df: pd.DataFrame):
        pass

    def transform(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        mappings = defaultdict(list)
        mapping_dict = defaultdict(dict)

        lhs_values = dirty_df[lhs_col].values.tolist()
        rhs_values = dirty_df[rhs_col].values.tolist()

        for index, l_value in enumerate(lhs_values):
            mappings[l_value].append(rhs_values[index])

        for l_value in mappings.keys():
            for r_value in mappings[l_value]:
                if len(mappings[l_value]) > 1:
                    mapping_dict[l_value][r_value] = 0
                else:
                    mapping_dict[l_value][r_value] = 1
        return np.asarray([mapping_dict[l_value][r_value] for l_value, r_value in zip(lhs_values, rhs_values)])

    def fit_transform_metal(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        return self.fit_transform(dirty_df, lhs_col, rhs_col)
