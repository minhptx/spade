import asyncio
import itertools
import random
import time
from abc import ABCMeta
from collections import Counter
from difflib import SequenceMatcher
from logging import error

import modin
import numpy as np
import pandas as pd
import regex as re
import swifter
from kbclean.datasets.dataset import Dataset
from loguru import logger
from strsimpy.jaccard import Jaccard

SOS = "^"
EOS = "$"


class TransformOp(metaclass=ABCMeta):
    def transform(self, str_value):
        pass

    def validate_target(self, str_value):
        pass


class InsertAt(TransformOp):
    def __init__(self, after_str, position):
        self.after_str = after_str
        self.position = position

    def transform(self, str_value):
        return str_value[: self.position] + self.after_str + str_value[self.position :]

    def validate(self, str_value):
        return len(str_value) > self.position

    def validate_target(self, str_value):
        if (
            len(str_value) > self.position + len(self.after_str)
            and str_value[self.position : self.position + len(self.after_str)]
            == self.after_str
        ):
            return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, InsertAt):
            return False
        return self.after_str == o.after_str and self.position == o.position

    def __str__(self) -> str:
        return f"InsertAt({self.after_str}, {self.position})"

    def __hash__(self) -> int:
        return hash(str(self))


class InsertAfter(TransformOp):
    def __init__(self, after_str, prev_char, position):
        self.after_str = after_str
        self.prev_char = prev_char
        self.position = position

    def transform(self, str_value):
        match = list(re.finditer(re.escape(self.prev_char), str_value))[self.position]
        return str_value[: match.start()] + self.after_str + str_value[match.start() :]

    def validate(self, str_value):
        return len(re.findall(re.escape(self.prev_char), str_value)) > self.position

    def validate_target(self, str_value):
        try:
            match = list(re.finditer(re.escape(self.prev_char), str_value))[
                self.position
            ]
        except IndexError as e:
            return False
        position = match.start()

        if (
            len(str_value) > position + len(self.after_str)
            and str_value[position : position + len(self.after_str)] == self.after_str
        ):
            return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, InsertAfter):
            return False
        return (
            self.after_str == o.after_str
            and self.position == o.position
            and self.prev_char == o.prev_char
        )

    def __str__(self) -> str:
        return f"InsertAfter({self.after_str}, {self.prev_char}, {self.position})"

    def __hash__(self) -> int:
        return hash(str(self))


class InsertBefore(TransformOp):
    def __init__(self, after_str, follow_char):
        self.after_str = after_str
        self.follow_char = follow_char

    def transform(self, str_value):
        match = list(re.finditer(re.escape(self.follow_char), str_value))[self.position]
        return str_value[: match.end()] + self.after_str + str_value[match.end() :]

    def validate(self, str_value):
        return len(re.findall(re.escape(self.follow_char), str_value)) > self.position

    def validate_target(self, str_value):
        try:
            match = list(re.finditer(re.escape(self.follow_char), str_value))[
                self.position
            ]
        except IndexError:
            return False
        position = match.start()

        if (
            position >= len(self.after_str)
            and str_value[position - len(self.after_str) : position] == self.after_str
        ):
            return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, InsertBefore):
            return False
        return self.after_str == o.after_str and self.follow_char == o.follow_char

    def __str__(self) -> str:
        return f"InsertBefore({self.after_str}, {self.follow_char}, {self.position})"

    def __hash__(self) -> int:
        return hash(str(self))


class ReplaceAt(TransformOp):
    def __init__(self, before_str, after_str, position):
        self.before_str = before_str
        self.after_str = after_str
        self.position = position

    def validate(self, str_value):
        return len(re.findall(re.escape(self.before_str), str_value)) > self.position

    def transform(self, str_value):
        match = list(re.finditer(re.escape(self.before_str), str_value))[self.position]
        return str_value[: match.start()] + self.after_str + str_value[match.end() :]

    def validate_target(self, str_value):
        if self.after_str in str_value:
            index = str_value.find(self.after_str)
            if str_value[:index].count(self.before_str) == self.position:
                return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ReplaceAt):
            return False
        return (
            self.before_str == o.before_str
            and self.after_str == o.after_str
            and self.position == o.position
        )

    def __str__(self) -> str:
        return f"ReplaceAt({self.before_str}, {self.after_str}, {self.position})"

    def __hash__(self) -> int:
        return hash(str(self))


class ReplaceAll(TransformOp):
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str

    def transform(self, str_value):
        return re.sub(re.escape(self.before_str), self.after_str, str_value)

    def validate(self, str_value):
        return self.before_str in str_value

    def validate_target(self, str_value):
        return self.after_str in str_value and self.before_str not in str_value

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ReplaceAll):
            return False
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __str__(self) -> str:
        return f"ReplaceAll({self.before_str}, {self.after_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class DeleteAll(TransformOp):
    def __init__(self, before_str):
        self.before_str = before_str

    def transform(self, str_value):
        return re.sub(re.escape(self.before_str), "", str_value)

    def validate(self, str_value):
        return self.before_str in str_value

    def validate_target(self, str_value):
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DeleteAll):
            return False
        return self.before_str == o.before_str

    def __str__(self) -> str:
        return f"DeleteAll({self.before_str})"

    def __hash__(self) -> int:
        return hash(str(self))


class DeleteAt(TransformOp):
    def __init__(self, before_str, position):
        self.before_str = before_str
        self.position = position

    def transform(self, str_value):
        match = list(re.finditer(re.escape(self.before_str), str_value))[self.position]
        return str_value[: match.start()] + str_value[match.end() :]

    def validate(self, str_value):
        return len(re.findall(re.escape(self.before_str), str_value)) > self.position

    def validate_target(self, str_value):
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DeleteAt):
            return False
        return self.before_str == o.before_str and self.position == o.position

    def __str__(self) -> str:
        return f"DeleteAt({self.before_str}, {self.position})"

    def __hash__(self) -> int:
        return hash(str(self))


def learn_transform(error_str, cleaned_str):
    s = SequenceMatcher(None, cleaned_str, error_str)
    transforms = s.get_opcodes()

    transform_ops = []

    for (tag, i1, i2, j1, j2) in transforms:
        if tag == "insert":
            transform_ops.append(InsertAt(error_str[j1:j2], i1))
            if i1 == 0:
                transform_ops.append(InsertAfter(error_str[j1:j2], SOS, 0))
            else:
                transform_ops.append(
                    InsertAfter(
                        error_str[j1:j2],
                        cleaned_str[i1 - 1],
                        len(
                            re.findall(re.escape(cleaned_str[i1 - 1]), cleaned_str[:i1])
                        ),
                    )
                )
        elif tag == "delete":
            transform_ops.append(
                DeleteAt(
                    cleaned_str[i1:i2],
                    len(re.findall(re.escape(cleaned_str[i1:i2]), cleaned_str[:i2])),
                )
            )

            transform_ops.append(DeleteAll(cleaned_str[i1:i2]))
        elif tag == "replace":
            transform_ops.append(
                ReplaceAt(
                    cleaned_str[i1:i2],
                    error_str[j1:j2],
                    len(re.findall(re.escape(cleaned_str[i1:i2]), cleaned_str[:i2])),
                )
            )

            transform_ops.append(ReplaceAll(cleaned_str[i1:i2], error_str[j1:j2]))

    return transform_ops


class Clean2ErrorGenerator:
    def __init__(self):
        self.rule2prob = None

    def fit(self, string_pairs):
        transforms = [learn_transform(*string_pair) for string_pair in string_pairs]

        transforms = [item for transform in transforms for item in transform]
        counter = Counter(transforms)
        sum_counter = sum(counter.values())
        self.rule2prob = {
            transform: count * 1.0 / sum_counter for transform, count in counter.items()
        }

        return self.rule2prob

    def _filter(self, dirty_df, col, pos_indices, neg_indices):
        def match_any(value):
            for rule in self.rule2prob.keys():
                if rule.validate_target(value):
                    return True
            return False

        dirty_df["matched"] = dirty_df[col].apply(lambda x: match_any(x))

        indices = dirty_df[dirty_df["matched"] == True].index.values.tolist()

        dirty_df.drop(["matched"], axis=1, inplace=True)

        return indices

    def sub_transform(self, row, col, pos_indices, neg_indices, pos_values):
        new_values = []
        new_labels = []
        new_rules = []

        rules = list(self.rule2prob.keys())

        if row.name in pos_indices + neg_indices:
            new_values.append(row[col])
            new_labels.append(row[f"{col}_labels"])
            if row.name in neg_indices:
                new_rules.append("Negative Populated Example (<0.5)")
            else:
                new_rules.append("Positive Populated Example (>0.5)")
            random.shuffle(rules)

            for rule in rules:
                if rule.validate(row[col]):
                    result = rule.transform(row[col])
                    if result not in pos_values:
                        new_values.append(result)
                        new_labels.append(0.0)
                        new_rules.append(str(rule))
                        break
        # elif row[f"{col}_weights"] > 0.5:
        #     new_values.append(row[col])
        #     new_labels.append(row[f"{col}_weights"])
        #     new_rules.append("PSL inference score")

        #     random.shuffle(rules)
        #     for rule in rules:
        #         if rule.validate(row[col]):
        #             result = rule.transform(row[col])
        #             if result not in pos_values:
        #                 new_values.append(result)
        #                 new_labels.append(0.0)
        #                 new_rules.append(str(rule))
        #                 break        
        row[col] = new_values
        row["new_labels"] = new_labels
        row["new_rules"] = new_rules
        return row

    def transform(self, dataset, col, pos_indices, neg_indices):
        new_dirty_df = dataset.dirty_df.copy()
        new_label_df = dataset.label_df.copy()
        new_rules = []

        pos_values = set([dataset.dirty_df[col][index] for index in pos_indices])

        positive = 0
        transformed_negative = 0

        row = dataset.dirty_df.iloc[0]
        label_row = dataset.label_df.iloc[0]

        new_dirty_df[f"{col}_labels"] = new_label_df[col]
        new_dirty_df[f"{col}_weights"] = dataset.soft_label_df[col]

        new_dirty_df = new_dirty_df.swifter.allow_dask_on_strings(True).apply(
            self.sub_transform,
            axis=1,
            raw=False,
            args=[col, pos_indices, neg_indices, pos_values],
        )
        new_dirty_df[col] = new_dirty_df[col].apply(lambda x: np.nan if len(x) == 0 else x)
        new_dirty_df = new_dirty_df.dropna()

        new_label_df = new_label_df.iloc[new_dirty_df.index.tolist()]

        new_label_df[col] = new_dirty_df["new_labels"]
        new_rules = new_dirty_df["new_rules"]

        new_dirty_df = new_dirty_df.drop(["new_labels", "new_rules", f"{col}_labels", f"{col}_weights"], axis=1)
        new_dirty_df = new_dirty_df.explode(col)
        new_label_df = new_label_df.explode(col)
        new_rules = [x for rules in new_rules for x in rules]

        for error_str, clean_str in dataset.col2labeled_pairs[col]:
            new_row = row.copy()
            new_row[col] = error_str
            new_groundtruth_row = label_row.copy()
            new_groundtruth_row[col] = int(clean_str == error_str)
            new_dirty_df = new_dirty_df.append(new_row, ignore_index=True)
            new_label_df = new_label_df.append(new_groundtruth_row, ignore_index=True)
            if clean_str != error_str:
                new_rules.append("False Example")
            else:
                new_rules.append("True Example")

            if clean_str != error_str:
                new_row = row.copy()
                new_row[col] = clean_str
                new_groundtruth_row = label_row.copy()
                new_groundtruth_row[col] = 1.0
                new_dirty_df = new_dirty_df.append(new_row, ignore_index=True)
                new_label_df = new_label_df.append(
                    new_groundtruth_row, ignore_index=True
                )
                new_rules.append("True Example")

        logger.info(
            f"Training set have size {len(new_dirty_df)} with {positive} positives and {transformed_negative} transformed negative"
        )
        return (
            new_dirty_df.reset_index(drop=True).sort_index().sort_index(axis=1),
            new_label_df.reset_index(drop=True).sort_index().sort_index(axis=1),
            new_rules,
        )

    def fit_transform(self, dataset, col: str, pos_indices, neg_indices):
        self.fit(dataset.col2labeled_pairs[col])

        return self.transform(dataset, col, pos_indices, neg_indices)


class ErrorGenerator:
    def __init__(self):
        self.clean2error = Clean2ErrorGenerator()

    def fit_transform(self, dataset, col: str, pos_indices, neg_indices):
        dataset = self.clean2error.fit_transform(dataset, col, pos_indices, neg_indices)

        return dataset
