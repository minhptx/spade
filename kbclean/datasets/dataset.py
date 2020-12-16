from kbclean.utils import data
from pathlib import Path
import pandas as pd


class Dataset:
    def __init__(self, dirty_df, clean_df):
        self.dirty_df = dirty_df
        self.clean_df = clean_df

        self.groundtruth_df = self.clean_df == self.dirty_df


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

        return Dataset(dirty_df, clean_df)

    def __add__(self, dataset: "Dataset"):
        dirty_df = pd.concat([self.dirty_df, dataset.dirty_df], ignore_index=True)
        clean_df = pd.concat([self.clean_df, dataset.clean_df], ignore_index=True)

        return Dataset(dirty_df, clean_df)


class LabeledDataset:
    def __init__(self, dirty_df, label_df):
        self.dirty_df = dirty_df
        self.label_df = label_df

    def __add__(self, dataset: "Dataset"):
        dirty_df = pd.concat([self.dirty_df, dataset.dirty_df], ignore_index=True)
        groundtruth_df = pd.concat([self.label_df, dataset.groundtruth_df], ignore_index=True)

        return Dataset(dirty_df, groundtruth_df)

    def filter_examples(self, col, label=True):
        dirty_df = self.dirty_df.copy()
        dirty_df["label"] = self.label_df[col]
        return dirty_df[dirty_df["label"] == label][col].values.tolist()
