import time

import numpy as np
import pandas as pd
from kbclean.detection.base import ActiveDetector
from kbclean.detection.features.base import ConcatFeaturizer
from kbclean.detection.features.embedding import CharAvgFT, WordAvgFT
from kbclean.detection.features.statistics import StatsFeaturizer, TfidfFeaturizer
from kbclean.transformation.error import ErrorGenerator
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



class RFDetector(ActiveDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = ConcatFeaturizer(
            {"char_ft": CharAvgFT(), "word_ft": WordAvgFT(), "stats": StatsFeaturizer()}
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

    def idetect_col(self, dataset, col, pos_indices, neg_indices):
        generator = ErrorGenerator()
        start_time = time.time()
        self.training = True

        logger.info("Transforming data ....")
        train_dirty_df, train_label_df, rules = generator.fit_transform(
            dataset, col, pos_indices, neg_indices
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

        feature_tensors = self.extract_features(dataset.dirty_df, None, col)

        pred = self.model.predict_proba(feature_tensors)

        logger.info(f"Total prediction time: {time.time() - start_time}")

        return pred[:, 1]

    def idetect(self, dataset: pd.DataFrame):
        for col_i, col in enumerate(dataset.dirty_df.columns):
            start_time = time.time()
            value_arr = dataset.label_df.iloc[:, col_i].values
            neg_indices = np.where(np.logical_and(0 <= value_arr, value_arr <= 0.5))[
                0
            ].tolist()
            pos_indices = np.where(value_arr >= 0.5)[0].tolist()
            examples = [dataset.dirty_df[col][i] for i in neg_indices + pos_indices]

            logger.info(
                f"Column {col} has {len(neg_indices)} negatives and {len(pos_indices)} positives"
            )

            print('Preparation time: ', time.time() - start_time)

            if len(neg_indices) == 0:
                dataset.prediction_df[col] = pd.Series(
                    [1.0 for _ in range(len(dataset.dirty_df))]
                )
            else:
                start_time = time.time()
                logger.info(f"Detecting column {col} with {len(examples)} examples")

                # pd.DataFrame(dataset.col2labeled_pairs[col]).to_csv(
                #     f"{self.hparams.debug_dir}/{col}_chosen.csv"
                # )

                outliers = self.idetect_col(dataset, col, pos_indices, neg_indices)

                df = pd.DataFrame(dataset.dirty_df[col].values)
                df["result"] = outliers
                df["training_label"] = dataset.label_df[col].values.tolist()
                df.to_csv(f"{self.hparams.debug_dir}/{col}_prediction.csv", index=None)
                dataset.prediction_df[col] = pd.Series(outliers)
                print('Training time: ', time.time() - start_time)


class XGBDetector(RFDetector):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.feature_extractor = ConcatFeaturizer(
            {"char_ft": CharAvgFT(), "stats": TfidfFeaturizer()}
        )
        self.model = XGBClassifier()
    

    def idetect_col(self, dataset, col, pos_indices, neg_indices):
        generator = ErrorGenerator()
        start_time = time.time()
        self.training = True

        logger.info("Transforming data ....")
        train_dirty_df, train_label_df, rules = generator.fit_transform(
            dataset, col, pos_indices, neg_indices
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

        self.model = XGBClassifier()
        self.model.fit(features, labels)

        feature_tensors = self.extract_features(dataset.dirty_df, None, col)

        pred = self.model.predict_proba(feature_tensors)

        logger.info(f"Total prediction time: {time.time() - start_time}")

        return pred[:, 1]