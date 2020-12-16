import itertools
from logging import error
import random
import time
from abc import ABCMeta
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import regex as re
from datasketch import MinHash, MinHashLSH
from kbclean.datasets.dataset import Dataset, LabeledDataset
from loguru import logger
from strsimpy.jaccard import Jaccard
import asyncio

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


async def learn_transform(error_str, cleaned_str):
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

    async def fit(self, string_pairs):
        transforms = await asyncio.gather(
            *(learn_transform(*string_pair) for string_pair in string_pairs)
        )

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


    def sub_transform(self, row, col, pos_indices, neg_indices, str_pairs):
        new_values = []
        new_labels = []
        new_rules = []

        rules = list(self.rule2prob.keys())


        if row.index in pos_indices + neg_indices:
            new_values.append(row[col])
            new_labels.append(row[f"{col}_values"])
            if row.index in neg_indices:
                new_rules.append("Negative Populated Example (<0.5)")
            else:
                new_rules.append("Positive Populated Example (>0.5)")
            random.shuffle(rules)

            for rule in rules:
                if rule.validate(row[col]):
                    new_row = row.copy()
                    result = rule.transform(row[col])
                    if result not in pos_values:
                        new_row[col] = result
                        new_dirty_df = new_dirty_df.append(
                            new_row, ignore_index=True
                        )
                        label_row = label_df.iloc[index].copy()
                        label_row[col] = 0.0
                        new_label_df = new_label_df.append(
                            label_row, ignore_index=True
                        )
                        new_rules.append(str(rule))
                        transformed_negative += 1
                        break

        else:
            new_dirty_df = new_dirty_df.append(row, ignore_index=True)
            label_row = label_df.iloc[index].copy()
            label_row[col] = 0.7
            new_label_df = new_label_df.append(label_row, ignore_index=True)
            new_rules.append("Not labeled example")

            random.shuffle(rules)
            for rule in rules:
                if rule.validate(row[col]):
                    new_row = row.copy()
                    result = rule.transform(row[col])
                    if result not in pos_values:
                        new_row[col] = result
                        new_dirty_df = new_dirty_df.append(
                            new_row, ignore_index=True
                        )
                        label_row = label_df.iloc[index].copy()
                        label_row[col] = 0.3
                        new_label_df = new_label_df.append(
                            label_row, ignore_index=True
                        )
                        new_rules.append(str(rule))
                        transformed_negative += 1
                        break

    def transform(self, dirty_df, label_df, col, pos_indices, neg_indices, str_pairs):
        new_dirty_df = pd.DataFrame(columns=dirty_df.columns, dtype=str)
        new_label_df = pd.DataFrame(columns=label_df.columns, dtype=str)
        new_rules = []

        # indices = self._filter(dirty_df, col, pos_indices, neg_indices)

        pos_values = set([dirty_df[col][index] for index in pos_indices])
        rules = list(self.rule2prob.keys())
        positive = 0
        transformed_negative = 0

        row = dirty_df.iloc[0]
        label_row = label_df.iloc[0]

        for error_str, clean_str in str_pairs:
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

        for index, row in dirty_df.iterrows():
            if len(new_dirty_df) > 5000:
                break
            label_row = label_df.iloc[index].copy()

            if index in pos_indices + neg_indices:
                new_dirty_df = new_dirty_df.append(row, ignore_index=True)
                new_label_df = new_label_df.append(label_row, ignore_index=True)
                if index in neg_indices:
                    new_rules.append("Negative Populated Example (<0.5)")
                else:
                    new_rules.append("Positive Populated Example (>0.5)")
                    positive += 1
                random.shuffle(rules)

                for rule in rules:
                    if rule.validate(row[col]):
                        new_row = row.copy()
                        result = rule.transform(row[col])
                        if result not in pos_values:
                            new_row[col] = result
                            new_dirty_df = new_dirty_df.append(
                                new_row, ignore_index=True
                            )
                            label_row = label_df.iloc[index].copy()
                            label_row[col] = 0.0
                            new_label_df = new_label_df.append(
                                label_row, ignore_index=True
                            )
                            new_rules.append(str(rule))
                            transformed_negative += 1
                            break

            else:
                new_dirty_df = new_dirty_df.append(row, ignore_index=True)
                label_row = label_df.iloc[index].copy()
                label_row[col] = 0.7
                new_label_df = new_label_df.append(label_row, ignore_index=True)
                new_rules.append("Not labeled example")

                random.shuffle(rules)
                for rule in rules:
                    if rule.validate(row[col]):
                        new_row = row.copy()
                        result = rule.transform(row[col])
                        if result not in pos_values:
                            new_row[col] = result
                            new_dirty_df = new_dirty_df.append(
                                new_row, ignore_index=True
                            )
                            label_row = label_df.iloc[index].copy()
                            label_row[col] = 0.3
                            new_label_df = new_label_df.append(
                                label_row, ignore_index=True
                            )
                            new_rules.append(str(rule))
                            transformed_negative += 1
                            break

        logger.info(
            f"Training set have size {len(new_dirty_df)} with {positive} positives and {transformed_negative} transformed negative"
        )
        return (
            new_dirty_df.reset_index(drop=True).sort_index().sort_index(axis=1),
            new_label_df.reset_index(drop=True).sort_index().sort_index(axis=1),
            new_rules,
        )

    def fit_transform(
        self, dirty_df, label_df, col: str, pos_indices, neg_indices, string_pairs
    ):
        asyncio.run(self.fit(string_pairs))

        return self.transform(
            dirty_df, label_df, col, pos_indices, neg_indices, string_pairs
        )


class SameClassGenerator:
    def __init__(self):
        self.rule2prob = {}

    def learn_transform(self, error_str, cleaned_str):
        s = SequenceMatcher(None, error_str, cleaned_str)
        transforms = s.get_opcodes()

        transform_ops = []

        for (tag, i1, i2, j1, j2) in transforms:
            if tag == "replace":
                transform_ops.append(
                    ReplaceAt(
                        cleaned_str[j1:j2],
                        error_str[i1:i2],
                        len(
                            re.findall(re.escape(cleaned_str[j1:j2]), cleaned_str[:j2])
                        ),
                    )
                )

                if cleaned_str.count(cleaned_str[j1:j2]) == 1:
                    transform_ops.append(
                        ReplaceAll(cleaned_str[j1:j2], error_str[i1:i2])
                    )

        return transform_ops

    def fit(self, string_pairs):
        transforms = []
        for error_str, cleaned_str in string_pairs:
            transforms.extend(self.learn_transform(error_str, cleaned_str))

        counter = Counter(transforms)
        sum_counter = sum(counter.values())
        self.rule2prob = {
            transform: count * 1.0 / sum_counter for transform, count in counter.items()
        }

        return self.rule2prob

    def transform(self, dataset, col, pos_indices, neg_indices):
        new_dirty_df = pd.DataFrame(columns=dataset.dirty_df.columns, dtype=str)
        new_clean_df = pd.DataFrame(columns=dataset.clean_df.columns, dtype=str)

        pos_values = [dataset.dirty_df[col][index] for index in pos_indices]

        for index in neg_indices:
            neg_row = dataset.dirty_df.iloc[index]
            pos_row = dataset.clean_df.iloc[index]
            for rule, _ in self.rule2prob.items():
                if rule.validate(neg_row[col]):
                    new_row = neg_row.copy()
                    result = rule.transform(neg_row[col])
                    new_row[col] = result
                    new_dirty_df = new_dirty_df.append(new_row, ignore_index=True)
                    new_clean_df = new_clean_df.append(pos_row, ignore_index=True)

            new_dirty_df = new_dirty_df.append(neg_row, ignore_index=True)
            new_clean_df = new_clean_df.append(pos_row, ignore_index=True)

        for index in pos_indices:
            pos_row = dataset.dirty_df.iloc[index]

            new_dirty_df = new_dirty_df.append(pos_row, ignore_index=True)
            new_clean_df = new_clean_df.append(pos_row, ignore_index=True)

        return Dataset(
            new_dirty_df.reset_index(drop=True).sort_index().sort_index(axis=1),
            new_clean_df.reset_index(drop=True).sort_index().sort_index(axis=1),
        )

    def fit_transform(
        self, dataset: Dataset, col: str, pos_indices, neg_indices, col2pairs
    ):
        neg_values = dataset.dirty_df.loc[neg_indices, col].values.tolist()
        pos_values = dataset.dirty_df.loc[pos_indices, col].values.tolist()
        jaccard = Jaccard(1)
        string_pairs = []
        count = 0
        if pos_values:
            while len(string_pairs) < 3000 and count < 10000:
                pos_value1 = random.choice(pos_values)
                pos_value2 = random.choice(pos_values)
                if (
                    pos_value1 != pos_value2
                    and jaccard.similarity(pos_value1, pos_value2) > 0.5
                ):
                    string_pairs.append((pos_value1, pos_value2))
                count += 1

        self.fit(string_pairs)

        return self.transform(dataset, col, pos_indices, neg_indices)


class ErrorGenerator:
    def __init__(self):
        self.clean2error = Clean2ErrorGenerator()
        self.same_class = SameClassGenerator()

    def fit_transform(
        self, dirty_df, label_df, col, pos_indices, neg_indices, string_pairs
    ):
        dataset1 = self.clean2error.fit_transform(
            dirty_df, label_df, col, pos_indices, neg_indices, string_pairs
        )

        return dataset1
