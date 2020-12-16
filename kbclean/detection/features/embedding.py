import pandas as pd
import torch
from kbclean.datasets.dataset import Dataset
from kbclean.detection.features.base import BaseFeaturizer
from kbclean.utils.inout import FastTextLoader
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer

fasttext = FastTextLoader.get_instance()


class CharSeqFT(BaseFeaturizer):
    def fit(self, dirty_df: pd.DataFrame, col: str):
        pass

    def transform(self, dirty_df: pd.DataFrame, col: str):
        result = stack_and_pad_tensors(
            [
                fasttext.get_vecs_by_tokens(list(val) + ["</s>"])
                for val in dirty_df[col].values
            ]
        )

        return [result.tensor, result.lengths]

    def feature_dim(self):
        return 2

    def n_features(self, dirty_df: pd.DataFrame):
        return 300


class CharAvgFT(BaseFeaturizer):
    def fit(self, dirty_df: pd.DataFrame, col: str):
        pass

    def transform(self, dirty_df: pd.DataFrame, col: str):
        return [
            torch.stack(
                [
                    torch.mean(fasttext.get_vecs_by_tokens(list(val) + ["</s>"]), dim=0)
                    for val in dirty_df[col].values
                ]
            )
        ]

    def feature_dim(self):
        return 1

    def n_features(self, dirty_df: pd.DataFrame):
        return 300


class WordSeqFT(BaseFeaturizer):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")

    def fit(self, dirty_df: pd.DataFrame, col: str):
        pass

    def transform(self, dirty_df: pd.DataFrame, col: str):
        result = stack_and_pad_tensors(
            [
                fasttext.get_vecs_by_tokens(self.tokenizer(val) + ["</s>"])
                for val in dirty_df[col].values
            ]
        )

        return [result.tensor, result.lengths]

    def feature_dim(self):
        return 2

    def n_features(self, dirty_df: pd.DataFrame):
        return 300


class WordAvgFT(BaseFeaturizer):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")

    def fit(self, dirty_df: pd.DataFrame, col: str):
        pass

    def transform(self, dirty_df: pd.DataFrame, col: str):
        return [
            torch.stack(
                [
                    torch.mean(
                        fasttext.get_vecs_by_tokens(self.tokenizer(val) + ["</s>"]),
                        dim=0,
                    )
                    for val in dirty_df[col].values
                ]
            )
        ]

    def feature_dim(self):
        return 1

    def n_features(self, dirty_df: pd.DataFrame):
        return 300


class CoValueAvgFT(BaseFeaturizer):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")

    def fit(self, dirty_df: pd.DataFrame, col: str):
        pass

    def transform(self, dirty_df: pd.DataFrame, col: str):
        join_sr = dirty_df.agg(" ".join, axis=1)

        return [
            torch.stack(
                [
                    torch.mean(fasttext.get_vecs_by_tokens(self.tokenizer(val)), dim=0,)
                    for val in join_sr.values
                ]
            )
        ]

    def n_features(self, dirty_df: pd.DataFrame):
        return 300

    def feature_dim(self):
        return 1
