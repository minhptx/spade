from kbclean.detection.features.embedding import (
    CharAvgFastText,
    CharFastText,
    WordAvgFastText,
)
from kbclean.detection.features.statistics import StatsExtractor
from kbclean.detection.features.base import ConcatExtractor
from typing import List

import numpy as np
import pandas as pd
from kbclean.detection.base import ActiveDetector
from kbclean.utils.data.readers import RowBasedValue
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


class RandomForestDetector(ActiveDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = RandomForestClassifier()

        self.feature_extractor = ConcatExtractor(
            {
                "char_ft": CharAvgFastText(),
                "word_ft": WordAvgFastText(),
                "stats": StatsExtractor(),
            }
        )

    def extract_features(self, data, labels=None, retrain=False):
        if labels:
            features = self.feature_extractor.fit_transform(data)
            self.hparams.model.feature_dim = self.feature_extractor[
                "stats"
            ].n_features()
        else:
            features = self.feature_extractor.transform(data)

        features = [feature.cpu().detach().numpy() for feature in features]

        if labels is not None:
            label_data = np.asarray(labels)
            return np.concatenate(features, axis=1), label_data
        return np.concatenate(features, axis=1)

    def reset(self):
        self.model = RandomForestClassifier()

    def eval_idetect(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, k):
        pass

    def idetect_values(
        self, ec_str_pairs: str, row_values: List[RowBasedValue], scores, active_learner
    ):
        data = [x[0] for x in ec_str_pairs]
        labels = [x[0] == x[1] for x in ec_str_pairs]

        positives = active_learner.most_positives(row_values, scores)

        features, labels = self.extract_features(
            data + positives,
            labels + [True for _ in range(len(positives))],
            retrain=True,
        )

        features, labels = SMOTE(k_neighbors=2).fit_resample(features, labels)

        self.model.fit(features, labels)

        features = self.extract_features(row_values)

        pred = self.model.predict_proba(features)

        return pred[:, 0]

    def idetect(self, df: pd.DataFrame, score_df: pd.DataFrame, recommender):
        result_df = df.copy()
        records = df.to_dict("records")
        for column in df.columns:
            values = df[column].values.tolist()
            row_values = [
                RowBasedValue(value, column, row_dict)
                for value, row_dict in zip(values, records)
            ]
            if score_df is not None:
                scores = score_df[column].values.tolist()
            else:
                scores = None
            if (
                column not in recommender.col2examples
                or not recommender.col2examples[column]
                or all(
                    [x[0].value == x[1].value for x in recommender.col2examples[column]]
                )
            ):
                result_df[column] = pd.Series([1.0 for _ in range(len(df))])
            else:
                outliers = self.idetect_values(
                    recommender.col2examples[column], row_values, scores, recommender,
                )
                result_df[column] = pd.Series(outliers)
        return result_df
