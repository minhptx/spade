import itertools
import time

import numpy as np
import pandas as pd
from kbclean.recommendation.checker.format import (
    CharFormatTHChecker,
    PunctFormatTHChecker,
    WordFormatTHChecker,
)
from kbclean.recommendation.checker.missing_value import MissingValueChecker
from kbclean.recommendation.checker.typo import (
    FastTextTHChecker,
    WebTableTHChecker,
    DictTypoTHChecker,
)
from kbclean.utils.data.attribute import xngrams
from kbclean.utils.search.query import ESQuery
from loguru import logger
from snorkel.labeling.model.label_model import LabelModel


def min_ngram_counts(es_query, values):
    trigrams = list(
        itertools.chain.from_iterable(
            [["".join(x) for x in xngrams(list(value), 3, False)] for value in values]
        )
    )
    counts = es_query.get_char_ngram_counts(trigrams)
    value2count = {trigram: count for trigram, count in zip(trigrams, counts)}

    return [
        min(
            [
                value2count[x]
                for x in ["".join(x) for x in xngrams(list(value), 3, False)]
            ]
        )
        for value in values
    ]


class MetalLeaner:
    def __init__(self, dirty_df, oracle, hparams):
        self.hparams = hparams

        self.dirty_df = dirty_df

        self.oracle = oracle

        self.es_query = ESQuery.get_instance(hparams.es_host, hparams.es_port)


        thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]
        thresholds1 = [1, 100, 1000, 10000, 100000]
        
        self.col2criteria = {
            col: [
                    (f"missing_values", MissingValueChecker(), [1]),
                    (f"fasttext_typo", FastTextTHChecker(), [1]),
                    (f"table_typo", WebTableTHChecker(self.es_query), thresholds1),
                    (f"char_format", CharFormatTHChecker(), thresholds),
                    (f"word_format", WordFormatTHChecker(), thresholds),
                    (f"punct_format", PunctFormatTHChecker(), thresholds),
                ]
            for col in self.dirty_df.columns
        }


        self.col2label_model = {
            col: LabelModel(cardinality=2) for col in self.dirty_df.columns
        }

        self.col2feature_df = {}

    def get_features(self, col, score_df, label_df):
        feature_arrs = []

        feature_df = pd.DataFrame({col: self.dirty_df[col].values.tolist()})

        for name, criterion, threshold in self.col2criteria[col]:
            for threshold in threshold:
                if all(label_df[col] == -1):
                    val_arr = criterion.fit_transform(self.dirty_df, col, threshold=threshold)
                else:
                    val_arr = criterion.transform(self.dirty_df, col, threshold=threshold)
                feature_df[f"{threshold}_{name}"] = val_arr

                feature_arrs.append(val_arr.reshape(-1, 1))
        feature_df.to_csv(f"{self.hparams.debug_dir}/{col}_feature.csv")
        feature_matrix = np.concatenate(feature_arrs, axis=1).astype(int)
        return feature_matrix

    def fit(self, col, score_df, label_df, col2pairs):
        feature_matrix = self.get_features(col, score_df, label_df)
        feature_df = pd.DataFrame(feature_matrix, dtype=str)
        feature_df["feature_str"] = feature_df.agg("|||".join, axis=1)
        self.col2label_model[col].fit(feature_matrix)
        self.col2feature_df[col] = feature_df

        return self.col2label_model[col].predict_proba(feature_matrix)[:, 1]

    def predict(self, probs, col, label_df, col2pairs, e):
        best_indices = []
        grouped_indices = []

        main_chosen_indices = label_df[
            (label_df[col] == 0) & (label_df[col] == 1)
        ].index
        chosen_indices = label_df[label_df[col] != -1].index

        probs[main_chosen_indices] += 90
        probs[chosen_indices] += 10

        for i in range(e):
            best_index = np.argmin(probs)
            best_indices.append(best_index)
            best_score = probs[best_index]

            probs[best_index] += 90

            label = self.oracle.answer(col, best_index)
            col2pairs[col].append(self.oracle.get_pairs(col, best_index))

            min_features = self.col2feature_df[col]["feature_str"][best_index]
            indices = self.col2feature_df[col][
                self.col2feature_df[col]["feature_str"] == min_features
            ].index.tolist()
            logger.debug(
                f"Example {i} of column {col}: {self.dirty_df[col][best_index]} with score {best_score} label {label}"
            )

            probs[indices] += 10
            grouped_indices.extend(indices)

            for row_i in grouped_indices:
                if self.dirty_df.loc[row_i, col] == self.dirty_df.loc[best_index, col]:
                    label_df.loc[row_i, col] = float(label)
                elif label_df.loc[row_i, col] == -1:
                    label_df.loc[row_i, col] = float(label)

            logger.debug(f"Total examples for column {col}: {len(grouped_indices)}")
        return label_df

    def fit_predict(self, col, score_df, label_df, col2pairs, e):

        if any(label_df[col] == -1):
            probs = self.fit(col, score_df, label_df, col2pairs)

            result_df = pd.DataFrame(probs, columns=["negative_scores"])
            result_df["str"] = self.dirty_df[col]
            result_df.to_csv(f"{self.hparams.debug_dir}/{col}_metal.csv", index=None)

            return self.predict(probs, col, label_df, col2pairs, e)
        else:
            probs = np.abs(score_df[col].values - 0.5)
            return self.predict(probs, col, label_df, col2pairs, e)

    def next_for_each_col(self, col, score_df, label_df, col2pairs, e):
        label_df = self.fit_predict(col, score_df, label_df, col2pairs, e)

        return label_df, col2pairs
