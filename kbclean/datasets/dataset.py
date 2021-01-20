from collections import defaultdict
from kbclean.utils import data
from pathlib import Path
import pandas as pd


class Dataset:
    def __init__(self, name, dirty_df, clean_df):
        self.name = name
        self.dirty_df = dirty_df
        self.clean_df = clean_df

        self.groundtruth_df = self.clean_df == self.dirty_df
        self.user_label_df = pd.DataFrame(
            -1, columns=self.dirty_df.columns, index=self.dirty_df.index
        )

        self.label_df = pd.DataFrame(
            -1, columns=self.dirty_df.columns, index=self.dirty_df.index
        )
        self.soft_label_df = pd.DataFrame(
            -1, columns=self.dirty_df.columns, index=self.dirty_df.index
        )

        self.prediction_df = pd.DataFrame(
            1, columns=self.dirty_df.columns, index=self.dirty_df.index
        )

        self.col2labeled_pairs = defaultdict(list)

    def filter_examples(self, col, label=True):
        dirty_df = self.dirty_df.copy()
        dirty_df["label"] = self.groundtruth_df[col]
        if label:
            return dirty_df[dirty_df["label"] >= 0.5][col].values.tolist()
        return dirty_df[dirty_df["label"] <= 0.5][col].values.tolist()

    @staticmethod
    def from_path(file_path):
        file_path = Path(file_path)
        clean_df = pd.read_csv(
            file_path / "clean.csv", header=0, dtype=str, keep_default_na=False
        )
        dirty_df = pd.read_csv(
            file_path / "dirty.csv", header=0, dtype=str, keep_default_na=False
        )

        return Dataset(file_path.name, dirty_df, clean_df)