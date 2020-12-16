from operator import index
from pathlib import Path
import shutil

from loguru import logger
from kbclean.recommendation.checker.missing_value import MissingValueChecker
from kbclean.recommendation.checker.format import (
    CharFormatChecker,
    CharChecker, PerplexityChecker,
    PunctFormatChecker,
    WordFormatChecker,
)
from kbclean.recommendation.checker.typo import (
    DictTypoChecker,
    FastTextChecker,
    WebTableChecker,
)
from kbclean.utils.search.query import ESQuery
from pslpython.model import Model
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from pslpython.partition import Partition
import pandas as pd
import numpy as np
import time


class PSLUtils:
    @staticmethod
    def create_model_from_config_dir(name, dir_path):
        dir_path = Path(dir_path)

        predicate_path = dir_path / "predicate.txt"
        rule_path = dir_path / "rule.txt"

        model = Model(name)

        for predicate in PSLUtils.read_predicates_from_file(predicate_path):
            model.add_predicate(predicate)

        for rule in PSLUtils.read_rules_from_file(rule_path):
            model.add_rule(rule)

        return model

    @staticmethod
    def read_predicates_from_file(predicate_path):
        for line in predicate_path.open("r").readlines():
            name, closed, size = line.split("/")

            yield Predicate(name, closed == "closed", int(size))

    @staticmethod
    def read_rules_from_file(predicate_path):
        for line in predicate_path.open("r").readlines():
            yield Rule(line)


class PSLearner:
    def __init__(self, dirty_df, oracle, hparams):
        self.hparams = hparams

        self.dirty_df = dirty_df

        self.oracle = oracle

        self.es_query = ESQuery.get_instance(hparams.es_host, hparams.es_port)
        self.data_path = hparams.psl_data_path

        thresholds = [
            ("one", 0.05),
            ("two", 0.1),
            ("three", 0.2),
            ("four", 0.3),
            ("five", 0.5),
            ("six", 1.0),
        ]
        thresholds1 = [
            ("one", 1),
            ("two", 10),
            ("three", 100),
            ("four", 1000),
            ("five", 10000),
            ("six", 100000),
        ]

        self.col2label_model = {
            col: PSLUtils.create_model_from_config_dir(f"{col}_0",hparams.psl_config_path)
            for col in self.dirty_df.columns
        }

        self.col2criteria = {
            col: [
                (f"missing_values", MissingValueChecker()),
                (f"fasttext_typo", FastTextChecker()),
                (f"table_typo", WebTableChecker(self.es_query)),
                (f"char_format", CharFormatChecker()),
                (f"word_format", WordFormatChecker()),
                (f"punct_format", PunctFormatChecker()),
                (f"perplexity", PerplexityChecker()),
            ]
            for col in self.dirty_df.columns
        }

        self.col2feature_df = {}

    def generate_feature_files(self, col, label_df, score_df):
        feature_df = pd.DataFrame(columns=["index", "signal", "value"])
        feature_arrs = []
        self.feature_names = []

        Path(f"{self.data_path}/{col}").mkdir(parents=True, exist_ok=True)

        for name, criterion in self.col2criteria[col]:
            val_arr = criterion.fit_transform(
                self.dirty_df, col, threshold=None
            )
            print(name)
            self.feature_names.append(f"{name}")
            new_feature_df = pd.DataFrame(columns=["index", "signal", "value"])
            new_feature_df["index"] = list(range(len(val_arr)))
            new_feature_df["signal"] = [
                f"{name}" for _ in range(len(val_arr))
            ]
            new_feature_df["value"] = 1 - val_arr.astype(float)

            feature_df = pd.concat([feature_df, new_feature_df], axis=0)

            feature_arrs.append(val_arr.reshape(-1, 1))

        feature_df.to_csv(
            f"{self.data_path}/{col}/has_signal.txt", header=None, sep="\t", index=None
        )

        feature_matrix = np.concatenate(feature_arrs, axis=1).astype(float)
        col_feature_df = pd.DataFrame(feature_matrix, dtype=str, columns=self.feature_names)
        col_feature_df["value"] = self.dirty_df[col]
        col_feature_df["feature_str"] = col_feature_df.agg("|||".join, axis=1)

        self.col2feature_df[col] = col_feature_df
        col_feature_df.to_csv(f"{self.hparams.debug_dir}/{col}_feature.csv", index=None)
        values = self.dirty_df[col].values.tolist()
    
        if label_df is not None:
            feature_df = pd.DataFrame(columns=["index", "label"])
            feature_df["index"] = list(range(len(self.dirty_df)))
            feature_df["label"] = label_df[col].apply(lambda x: "positive" if x == 1 else "negative" if x == 0 else "n/a")
            feature_df["value"] = [1 for _ in range(len(feature_df))]
            feature_df.to_csv(
                f"{self.data_path}/{col}/has_label.txt", header=None, sep="\t", index=None
            )
        else:
            feature_df = pd.DataFrame(columns=["index", "label"])
            feature_df["index"] = list(range(len(self.dirty_df)))
            feature_df["label"] = ["n/a" for _ in range(len(feature_df))]
            feature_df.to_csv(
                f"{self.data_path}/{col}/has_label.txt", header=None, sep="\t", index=None
            )

        pd.DataFrame([f"{i}" for i in range(len(self.dirty_df))]).to_csv(
            f"{self.data_path}/{col}/target.txt", header=None, index=None
        )
        pd.DataFrame(self.feature_names).to_csv(
            f"{self.data_path}/{col}/good_signal.txt", header=None, index=None
        )

    def fit(self, col, score_df, label_df, col2pairs):
        self.generate_feature_files(col, score_df, label_df)

        self.col2label_model[col] = PSLUtils.create_model_from_config_dir("{col}_{time.time()}", self.hparams.psl_config_path)
        for file_path in (Path(self.data_path) / col).iterdir():
            if "target" in file_path.name or "good_signal" in file_path.name:
                self.col2label_model[col].get_predicate(file_path.stem).add_data_file(
                    Partition.TARGETS, str(file_path)
                )
            else:
                self.col2label_model[col].get_predicate(file_path.stem).add_data_file(
                    Partition.OBSERVATIONS, str(file_path)
                )

        result = self.col2label_model[col].infer(cleanup_temp=True)
        target_result = result["TARGET"]
        target_result = target_result.sort_values(by=[0])
        target_result["value"] = target_result[0].apply(lambda x: self.dirty_df.loc[int(x), col])
        target_result.to_csv(f"{self.hparams.debug_dir}/{col}_psl.csv", index=None)

        signal = result["GOOD_SIGNAL"]
        signal.to_csv(f"{self.hparams.debug_dir}/{col}_signal.csv")

        return target_result.loc[:, "truth"].values

    def predict(self, probs, col, label_df, col2pairs, e):
        best_indices = []
        grouped_indices = []

        main_chosen_indices = label_df[
            (label_df[col] == 0) & (label_df[col] == 1)
        ].index
        chosen_indices = label_df[label_df[col] != -1].index

        probs[main_chosen_indices] -= 90
        probs[chosen_indices] -= 10

        for i in range(e):
            best_index = np.argmax(probs)
            best_indices.append(best_index)
            best_score = probs[best_index]

            probs[best_index] -= 90

            label = self.oracle.answer(col, best_index)
            col2pairs[col].append(self.oracle.get_pairs(col, best_index))

            min_features = self.col2feature_df[col]["feature_str"][best_index]
            indices = self.col2feature_df[col][
                self.col2feature_df[col]["feature_str"] == min_features
            ].index.tolist()
            logger.debug(
                f"Example {i} of column {col}: {self.dirty_df[col][best_index]} with score {best_score} label {label}"
            )

            probs[indices] -= 10
            grouped_indices.extend(indices)

            for row_i in grouped_indices:
                if row_i == best_index:
                    label_df.loc[row_i, col] = float(label)
                elif (
                    self.dirty_df.loc[row_i, col] == self.dirty_df.loc[best_index, col]
                ):
                    label_df.loc[row_i, col] = float(label)
                # elif label_df.loc[row_i, col] == -1:
                #     if label == 1:
                #         label_df.loc[row_i, col] = 1.0
                #     else:
                #         label_df.loc[row_i, col] = 0.0

        return label_df

    def fit_predict(self, col, score_df, label_df, col2pairs, e):

        probs = self.fit(col, score_df, label_df, col2pairs)

        result_df = pd.DataFrame(probs, columns=["negative_scores"])
        result_df["str"] = self.dirty_df[col]
        result_df.to_csv(f"{self.hparams.debug_dir}/{col}_metal.csv", index=None)

        return self.predict(probs, col, label_df, col2pairs, e)

    def next_for_each_col(self, col, score_df, label_df, col2pairs, e):
        label_df = self.fit_predict(col, score_df, label_df, col2pairs, e)

        return label_df, col2pairs
