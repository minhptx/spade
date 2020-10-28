import itertools
from functools import partial
from typing import Counter, List

import numpy as np
import pandas as pd
import torch.nn.functional as F
from kbclean.detection.base import ActiveDetector, BaseModule
from kbclean.detection.extras.highway import Highway
from kbclean.transformation.noisy_channel import NCGenerator
from kbclean.utils.data.helpers import (
    str2regex,
)
from kbclean.utils.features.attribute import (
    sym_trigrams,
    sym_value_freq,
    val_trigrams,
    value_freq,
    xngrams,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from loguru import logger

class RFFeatureExtractor:
    def fit(self, values):
        logger.debug("Values: " + str(values[:10]))
        trigram = [["".join(x) for x in list(xngrams(val, 3))] for val in values]
        ngrams = list(itertools.chain.from_iterable(trigram))
        self.trigram_counter = Counter(ngrams)
        sym_ngrams = [str2regex(x, False) for x in ngrams]

        self.sym_trigram_counter = Counter(sym_ngrams)
        self.val_counter = Counter(values)

        sym_values = [str2regex(x, False) for x in values]
        self.sym_val_counter = Counter(sym_values)

        self.func2counter = {
            val_trigrams: self.trigram_counter,
            sym_trigrams: self.sym_trigram_counter,
            value_freq: self.val_counter,
            sym_value_freq: self.sym_val_counter,
        }

    def transform(self, values):
        feature_lists = []
        for func, counter in self.func2counter.items():
            f = partial(func, counter=counter)
            logger.debug(
                "Negative: %s %s" % (func, list(zip(values[:10], f(values[:10]))))
            ) 
            logger.debug(
                "Positive: %s %s" % (func, list(zip(values[-10:], f(values[-10:]))))
            )
            feature_lists.append(f(values))

        feature_vecs = list(zip(*feature_lists))
        return np.asarray(feature_vecs)


class RFDetector(ActiveDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = RFFeatureExtractor()

        self.model = RandomForestClassifier()
        self.scaler = MinMaxScaler()
        self.generator = NCGenerator()

    def extract_features(self, data, labels=None):
        if labels:
            features = self.scaler.fit_transform(self.feature_extractor.transform(data))
        else:
            features = self.scaler.transform(self.feature_extractor.transform(data))

        if labels is not None:
            return features, np.asarray(labels)
        return features
    
    def reset(self):
        self.model = RandomForestClassifier()

    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        self.feature_extractor.fit(values)

        data, labels = self.generator.fit_transform(ec_str_pairs, values)

        features, labels = self.extract_features(data, labels)

        self.model.fit(features, labels)

        test_features = self.extract_features(values)

        return self.model.predict_proba(test_features)[:, 1]

    def eval_idetect(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, k):
        result_df = raw_df.copy()
        for column in raw_df.columns:
            values = raw_df[column].values.tolist()
            cleaned_values = cleaned_df[column].values.tolist()
            false_values = []

            for val, cleaned_val in zip(values, cleaned_values):
                if val != cleaned_val:
                    false_values.append((val, cleaned_val))

            if not false_values:
                result_df[column] = pd.Series([True for _ in range(len(raw_df))])
            else:
                outliers = self.idetect_values(false_values[:k], values)
                result_df[column] = pd.Series(outliers)
        return result_df

    def idetect(self, df: pd.DataFrame, col2examples: dict):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            if column not in col2examples or not col2examples[column]:
                result_df[column] = pd.Series([1.0 for _ in range(len(df))])
            else:
                outliers = self.idetect_values(
                    [(x["raw"], x["cleaned"]) for x in col2examples[column]], values
                )
                result_df[column] = pd.Series(outliers)
        return result_df
