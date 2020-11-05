from typing import List
from kbclean.utils.data.readers import RowBasedValue
import numpy as np
import torch
from kbclean.detection.features.base import BaseExtractor
from kbclean.utils.data.helpers import str2regex
from sklearn.feature_extraction.text import CountVectorizer



class StatsExtractor(BaseExtractor):
    def __init__(self):
        self.char_counter = CountVectorizer(analyzer="char", lowercase=False)
        self.word_counter = CountVectorizer(lowercase=False)
        self.regex_counter = CountVectorizer(analyzer="char", lowercase=False)

        self.name2covalue_counter = {}

    def fit(self, values: List[RowBasedValue]):
        self.char_counter.fit([val.value for val in values])
        self.word_counter.fit([val.value for val in values])
        self.regex_counter.fit(
            [str2regex(val.value, match_whole_token=False) for val in values]
        )

        for name in values[0].row_dict.keys():
            if name == values[0].column_name:
                continue 
            covalue_list = []
            for value in values:
                covalue_list.append(f"{value}||{name}||{value.row_dict[name]}")
            self.name2covalue_counter[name] = CountVectorizer(
                analyzer=lambda x: [x]
            ).fit(covalue_list)

    def transform(self, values):
        char_features = self.char_counter.transform(
            [val.value for val in values]
        ).todense()
        word_features = self.word_counter.transform(
            [val.value for val in values]
        ).todense()
        regex_features = self.regex_counter.transform(
            [str2regex(val.value, match_whole_token=False) for val in values]
        ).todense()

        co_feature_lists = []
        for name in values[0].row_dict.keys():
            if name == values[0].column_name:
                continue
            covalue_list = []
            for value in values:
                covalue_list.append(f"{value}||{name}||{value.row_dict[name]}")
            co_feature_lists.append(
                self.name2covalue_counter[name].transform(covalue_list)
            )

        return [torch.tensor(
            np.concatenate(
                [char_features, regex_features],
                axis=1,
            )
        )]

    def n_features(self):
        return sum(
            [
                len(x.get_feature_names())
                for x in [self.char_counter, self.regex_counter]
            ]
        )

