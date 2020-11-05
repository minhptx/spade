from typing import List

import torch
from kbclean.detection.features.base import BaseExtractor
from kbclean.utils.data.readers import RowBasedValue
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import FastText

fasttext = FastText()


class CharFastText(BaseExtractor):
    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        return stack_and_pad_tensors(
            [
                fasttext.get_vecs_by_tokens(list(x.value) + ["</s>"])
                for x in values
            ]
        )

    def n_features(self):
        return 300


class CharAvgFastText(BaseExtractor):
    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        return [torch.stack(
            [
                torch.mean(fasttext.get_vecs_by_tokens(list(x.value) + ["</s>"]), dim=0)
                for x in values
            ]
        )]

    def n_features(self):
        return 300


class WordFastText(BaseExtractor):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")

    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        return stack_and_pad_tensors(
            [fasttext.get_vecs_by_tokens(self.tokenizer(x.value) + ["</s>"]) for x in values]
        )

    def n_features(self):
        return 300


class WordAvgFastText(BaseExtractor):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")

    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        return [torch.stack(
            [
                torch.mean(fasttext.get_vecs_by_tokens(self.tokenizer(x.value) + ["</s>"]), dim=0)
                for x in values
            ]
        )]

    def n_features(self):
        return 300

class CoValueFastText(BaseExtractor):
    def __init__(self):
        self.tokenizer = get_tokenizer("spacy")

    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        return torch.stack(
            [
                torch.mean(
                    fasttext.get_vecs_by_tokens(
                        self.tokenizer(list(x.row_dict.values()))
                    ),
                    dim=1,
                )
                for x in values
            ]
        )

    def n_features(self):
        return 300

