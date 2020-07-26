from typing import List

import pandas as pd
import regex as re
import spacy
from nltk.util import trigrams
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer

from kbclean.detection import BaseDetector
from kbclean.utils.data.helpers import str2regex

regex_dict = {
    "digit": r"[-+]?[0-9]+",
    "lower": r"[a-z]+",
    "upper": r"[A-Z]+",
    "whitespace": r" ",
}


def _default_featurize(str_):
    feature_dict = {}
    count = 0
    while str_:
        count += 1
        for name, pattern in regex_dict.items():
            match = re.match(f"^{pattern}", str_)
            if match:
                feature_dict[f"{name}_{count}"] = 1
                str_ = str_[match.end() :]
                break
        else:
            feature_dict[f"{str_[0]}_{count}"] = 1
            str_ = str_[1:]
    return feature_dict


def _ngram_featurize(str_):
    feature_dict = {}
    for trigram in trigrams(str_):
        feature_dict[f"{''.join(trigram)}"] = 1

    for trigram in trigrams(str2regex(str_)):
        feature_dict[f"pattern_{''.join(trigram)}"] = 1

    return feature_dict


class StatsDetector(BaseDetector):
    def __init__(self, hparams, featurize_func=_ngram_featurize):
        self.hparams = hparams

        self.vectorizer = DictVectorizer()
        self.outlier_detector = IsolationForest()
        self.featurize_func = featurize_func

        self.word2vec = spacy.load("en_core_web_lg")
        pattern2count = pd.read_csv(
            f"{self.hparams.save_path}/pattern2count.csv", error_bad_lines=False
        )
        self.pattern2count = dict(zip(pattern2count.pattern, pattern2count["count"]))

    def prepare(self):
        pass

    def featurize(self, values):
        features = map(self.featurize_func, values)
        return features

    def detect_outliers(self, values: List[str]):
        # if len(values) < 10:
        #     return [False for _ in range(len(values))]
        feature_dicts = map(self.featurize_func, values)
        feature_vecs = self.vectorizer.fit_transform(feature_dicts)
        outliers = self.outlier_detector.fit_predict(feature_vecs)
        outlier_results = []

        for outlier in outliers:
            if outlier == -1:
                outlier_results.append(False)
            else:
                outlier_results.append(True)
        return [False if outlier == -1 else True for outlier in outliers]

    def detect_null(self, values: List[str]):
        vectors = map(lambda x: self.word2vec(x).vector, values)
        null_results = []
        for index, vector in enumerate(vectors):
            if all(vector == 0):
                null_results.append(False)
            else:
                null_results.append(True)
        return null_results

    def detect_errors(self, values: List[str]):
        patterns = map(str2regex, values)
        error_results = []
        for index, pattern in enumerate(patterns):
            if self.pattern2count.get(pattern, 0) < 10:
                error_results.append(False)
            else:
                error_results.append(True)
        return error_results

    def detect_values(self, values):
        # errors = self.detect_outliers(values)
        outliers = self.detect_outliers(values)
        # nulls = self.detect_null(values)
        return [x for x in outliers]

    def detect(self, df):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            outliers = self.detect_values(values)
            result_df[column] = pd.Series(outliers)
        return result_df
