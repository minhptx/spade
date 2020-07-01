from typing import List
import pandas as pd
import regex as re
import spacy
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer

regex_dict = {
    "digit": r"[-+]?[0-9]+",
    "lower": r"[a-z]+",
    "upper": r"[A-Z]+",
    "whitespace": r" ",
}


def _to_regex(x):
    try:
        if x is None:
            return ""
        x = re.sub(r"[A-Z]+", "A", x)
        x = re.sub(r"[0-9]+", "0", x)
        x = re.sub(r"[a-z]+", "a", x)
        return x
    except Exception as e:
        print(e)
        return x


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


class StatsDetector:
    def __init__(
            self, featurize_func=_default_featurize, pattern2count_path="data/pattern.csv"
    ):
        self.vectorizer = DictVectorizer()
        self.outlier_detector = IsolationForest()
        self.featurize_func = featurize_func

        self.word2vec = spacy.load("en_core_web_lg")
        pattern2count = pd.read_csv(pattern2count_path)
        self.pattern2count = dict(zip(pattern2count.pattern, pattern2count["count"]))

    def featurize(self, values):
        features = map(self.featurize_func, values)
        return features

    def detect_table(self, df, output_path=None):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            errors = self.detect_errors(values)
            nulls = self.detect_null(values)
            # outliers = self.detect_outliers(values)
            result_df[column] = pd.Series(
                zip(values, list(zip(errors, nulls))), index=result_df.index
            ).apply(lambda x: f"[[[{x[0]}]]]" if any(x[1]) else f"{x[0]}")
        if output_path is not None:
            result_df.to_csv(output_path)
        return result_df

    def detect_outliers(self, values: List[str]):
        feature_dicts = map(self.featurize_func, values)
        feature_vecs = self.vectorizer.fit_transform(feature_dicts)
        outliers = self.outlier_detector.fit_predict(feature_vecs)
        outlier_results = []

        for index, outlier in enumerate(outliers):
            if outlier == -1:
                outlier_results.append(True)
            else:
                outlier_results.append(False)
        return outlier_results

    def detect_null(self, values: List[str]):
        vectors = map(lambda x: self.word2vec(x).vector, values)
        null_results = []
        for index, vector in enumerate(vectors):
            if all(vector == 0):
                null_results.append(True)
            else:
                null_results.append(False)
        return null_results

    def detect_errors(self, values: List[str]):
        patterns = map(_to_regex, values)
        error_results = []
        for index, pattern in enumerate(patterns):
            if self.pattern2count.get(pattern, 0) < 40:
                error_results.append(True)
            else:
                error_results.append(False)
        return error_results
