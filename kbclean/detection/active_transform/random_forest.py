import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from kbclean.datasets.dataset import Dataset
from kbclean.detection.active_transform.holo import HoloDetector
from kbclean.detection.features.base import ConcatFeaturizer, UnionFeaturizer
from kbclean.detection.features.embedding import CharAvgFT, WordAvgFT
from kbclean.detection.features.statistics import StatsFeaturizer
from kbclean.transformation.error import Clean2ErrorGenerator, ErrorGenerator
from kbclean.utils.data.helpers import split_train_test_dls, unzip_and_stack_tensors
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

class RFDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = ConcatFeaturizer(
            {"stats": StatsFeaturizer()}
        )
        self.training = True
        self.model = RandomForestClassifier(class_weight="balanced")

    def extract_features(self, dirty_df, label_df, col):
        if self.training:
            features = self.feature_extractor.fit_transform(dirty_df, col)
        else:
            features = self.feature_extractor.transform(dirty_df, col)

        features = features.detach().cpu().numpy()

        if self.training:
            labels = label_df[col]
            return features, np.asarray(labels.values >= 0.5)
        return features

    def reset(self):
        try:
            self.model = RandomForestClassifier()
        except:
            pass

    def idetect_col(self, dirty_df, label_df, col, pos_indices, neg_indices, pairs):
        generator = ErrorGenerator()
        start_time = time.time()
        self.training = True
        train_dirty_df, train_label_df, rules = generator.fit_transform(
            dirty_df, label_df, col, pos_indices, neg_indices, pairs
        )

        output_pd = train_dirty_df[[col]].copy()
        output_pd["label"] = train_label_df[col].values.tolist()
        output_pd["rule"] = rules
        output_pd.to_csv(f"{self.hparams.debug_dir}/{col}_debug.csv")

        logger.info(f"Total transformation time: {time.time() - start_time}")

        features, labels = self.extract_features(
            train_dirty_df, train_label_df, col
        )

        start_time = time.time()
        self.training = False

        self.model = RandomForestClassifier()
        self.model.fit(features, labels)

        feature_tensors = self.extract_features(dirty_df, label_df, col)

        pred = self.model.predict_proba(feature_tensors)

        logger.info(f"Total prediction time: {time.time() - start_time}")

        return pred[:, 1]

    def idetect(
        self, dirty_df: pd.DataFrame, label_df: pd.DataFrame, col2pairs,
    ):
        prediction_df = dirty_df.copy()

        for col_i, col in enumerate(dirty_df.columns):
            value_arr = label_df.iloc[:, col_i].values
            neg_indices = np.where(np.logical_and(0 <= value_arr, value_arr <= 0.5))[
                0
            ].tolist()
            pos_indices = np.where(value_arr >= 0.5)[0].tolist()
            examples = [dirty_df[col][i] for i in neg_indices + pos_indices]

            logger.info(
                f"Column {col} has {len(neg_indices)} negatives and {len(pos_indices)} positives"
            )

            if len(neg_indices) == 0:
                logger.info(
                    f"Skipping column {col} with {len(examples)} examples {list(set(examples))[:20]}"
                )
                prediction_df[col] = pd.Series([1.0 for _ in range(len(dirty_df))])
            else:
                logger.info(f"Detecting column {col} with {len(examples)} examples")

                pd.DataFrame(col2pairs[col]).to_csv(f"{self.hparams.debug_dir}/{col}_chosen.csv")

                logger.debug(
                    f"{len(dirty_df[label_df[col] >= 0.5])} Positive values: {dirty_df[label_df[col] >= 0.5][col].values.tolist()[:20]}"
                )
                logger.debug(
                    f"{len(dirty_df[label_df[col] <= 0.5])} Negative values: {dirty_df[label_df[col] <= 0.5][col].values.tolist()[:20]}"
                )
                outliers = self.idetect_col(
                    dirty_df, label_df, col, pos_indices, neg_indices, col2pairs[col],
                )

                df = pd.DataFrame(dirty_df[col].values)
                df["result"] = outliers
                df["training_label"] = label_df[col].values.tolist()
                df.to_csv(f"{self.hparams.debug_dir}/{col}_prediction.csv", index=None)
                prediction_df[col] = pd.Series(outliers)

        return prediction_df
