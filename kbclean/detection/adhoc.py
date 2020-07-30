from kbclean.utils.es.query import ESQuery
from typing import List

import pandas as pd
import regex as re
import spacy
from nltk.util import ngrams, trigrams
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


def _ngram_featurize(str_):
    feature_dict = {}
    if len(str_) <= 3:
        feature_dict[f"{str_}"] = 1
        feature_dict[f"{str2regex(str_)}"] = 1

    for trigram in trigrams(str_):
        feature_dict[f"{''.join(trigram)}"] = 1

    for trigram in trigrams(str2regex(str_)):
        feature_dict[f"pattern_{''.join(trigram)}"] = 1

    return feature_dict


class NgramChecker:
    def __init__(self, host):
        self.es_query = ESQuery.get_instance(host)
        self.ngram2threshold = {2: 100, 3: 10, 4: 5}

    @staticmethod
    def get_ngrams(str, n):
        try:
            return ngrams(list(str), n)
        except:
            return [str]

    def check_str(self, str):
        for i in [2, 3, 4]:
            ngrams = NgramChecker.get_ngrams(str)
            for ngram in ngrams:
                if ESQuery.get_char_ngram_count(ngram) < self.ngram2threshold[i]:
                    return False
        return True


class AdhocDetector(BaseDetector):
    def __init__(self, host, featurize_func=_ngram_featurize):

        self.vectorizer = DictVectorizer()
        self.outlier_detector = IsolationForest()
        self.featurize_func = featurize_func

        self.ngram_checker = NgramChecker(host)

        self.word2vec = spacy.load("en_core_web_lg")

    def prepare(self):
        pass

    def featurize(self, values):
        features = map(self.featurize_func, values)
        return features

    def save(self, save_path):
        pass

    def detect_outliers(self, values: List[str]):
        # if len(values) < 10:
        #     return [False for _ in range(len(values))]
        feature_dicts = list(map(self.featurize_func, values))
        print(values, feature_dicts)
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
        # patterns = map(str2regex, values)
        error_results = []
        for index, value in enumerate(values):
            if self.ngram_checker(value):
                error_results.append(False)
            else:
                error_results.append(True)
        return error_results

    def detect_values(self, values):
        errors = self.detect_outliers(values)
        outliers = self.detect_outliers(values)
        nulls = self.detect_null(values)
        return [all(x) for x in zip(errors, nulls)]

    def detect(self, df):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            outliers = self.detect_values(values)
            result_df[column] = pd.Series(outliers)
        return result_df
