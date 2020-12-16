import itertools
import random
from collections import Counter
from difflib import SequenceMatcher
import time

import numpy as np
import regex as re
from sklearn.naive_bayes import MultinomialNB
from kbclean.utils.data.attribute import xngrams
from loguru import logger
from kbclean.datasets.dataset import Dataset
import pandas as pd


def wskeep_tokenize(s):
    return re.split(r"([\W\p{P}])", s)


class CharTransform:
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str

    def transform(self, str_value):
        if not self.before_str:
            if not str_value:
                return self.after_str
            else:
                sample_position = random.randrange(len(str_value))
                return (
                    str_value[:sample_position]
                    + self.after_str
                    + str_value[sample_position:]
                )

        return str_value.replace(self.before_str, self.after_str)

    def __eq__(self, o: "CharTransform"):
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __hash__(self) -> int:
        return hash(f"Rule('{self.before_str}', '{self.after_str}')")

    def validate(self, str_value):
        return self.before_str in str_value

    def __repr__(self) -> str:
        return f"CharTransform('{self.before_str}', '{self.after_str}')"


class WordTransform:
    def __init__(self, before_str, after_str):
        self.before_str = before_str
        self.after_str = after_str

    def transform(self, str_value):
        if not self.before_str:
            if not str_value:
                return self.after_str
            else:
                positions = [(0, 0), (len(str_value), len(str_value))] + [
                    m.span() for m in re.finditer("[\p{P}\p{S}]", str_value)
                ]
                idx = random.randrange(len(positions))
                return (
                    str_value[: positions[idx][0]]
                    + self.after_str
                    + str_value[positions[idx][1] :]
                )
        return str_value.replace(self.before_str, self.after_str)

    def __eq__(self, o: "CharTransform"):
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __hash__(self) -> int:
        return hash(f"WordTransform('{self.before_str}', '{self.after_str}')")

    def validate(self, str_value):
        return self.before_str in str_value

    def __repr__(self) -> str:
        return f"Rule('{self.before_str}', '{self.after_str}')"


class CharNoisyChannel:
    def __init__(self):
        self.rule2prob = None

    def longest_common_substring(self, str1, str2):
        seqMatch = SequenceMatcher(None, str1, str2)

        match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

        if match.size != 0:
            return match.a, match.b, match.size
        else:
            return None

    def similarity(self, str1, str2):
        counter1 = Counter(list(str1))
        counter2 = Counter(list(str2))

        c = counter1 & counter2

        n = sum(c.values())
        try:
            return 2 * n / (len(str1) + len(str2))
        except ZeroDivisionError:
            return 0

    def learn_transformation(self, error_str, cleaned_str):
        if not cleaned_str and not error_str:
            return []

        valid_trans = [CharTransform(cleaned_str, error_str)]

        l = self.longest_common_substring(cleaned_str, error_str)

        if l is None:
            return valid_trans

        lcv, rcv = cleaned_str[: l[0]], cleaned_str[l[0] + l[2] :]
        lev, rev = error_str[: l[1]], error_str[l[1] + l[2] :]

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) >= self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or lev:
                valid_trans.append(CharTransform(lcv, lev))
            if rcv or rev:
                valid_trans.append(CharTransform(rcv, rev))
            valid_trans.extend(self.learn_transformation(lev, lcv))
            valid_trans.extend(self.learn_transformation(rev, rcv))

        elif self.similarity(lcv, lev) + self.similarity(rcv, rev) < self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or rev:
                valid_trans.append(CharTransform(lcv, rev))
            if rcv or lev:
                valid_trans.append(CharTransform(rcv, lev))
            valid_trans.extend(self.learn_transformation(rev, lcv))
            valid_trans.extend(self.learn_transformation(lev, rcv))

        return list(set(valid_trans))

    def fit(self, string_pairs):
        transforms = []
        for error_str, cleaned_str in string_pairs:
            transforms.extend(self.learn_transformation(error_str, cleaned_str))

        counter = Counter(transforms)
        sum_counter = sum(counter.values())
        self.rule2prob = {
            transform: count * 1.0 / sum_counter for transform, count in counter.items()
        }

        return self.rule2prob

    def get_exceptions(self):
        rule_values = list([x.after_str for x in self.rule2prob.keys()])
        one_grams = Counter(itertools.chain.from_iterable(rule_values))
        two_ngrams = Counter(
            itertools.chain.from_iterable(
                [
                    "".join(x)
                    for val in rule_values
                    for x in xngrams(val, 2, add_regex=False)
                ]
            )
        )
        return list(set(list(one_grams.keys()) + list(two_ngrams.keys())))

    def get_warning_substrs(self):
        return [x.after_str for x in self.rule2prob.keys() if not x.before_str]

    def _filter(self, dataset: Dataset, col, pos_indices, neg_indices):
        def match_any(substrs, value):
            for substr in substrs:
                if substr in value:
                    return True
            return False

        substrs = self.get_warning_substrs()
        dataset.dirty_df["matched"] = dataset.dirty_df[col].apply(lambda x: match_any(substrs, x))

        indices = dataset.dirty_df[dataset.dirty_df["matched"] == True].index.values.tolist()
        
        dataset.dirty_df.drop(["matched"], axis=1, inplace=True)

        return indices


    def transform(self, dataset, col, pos_indices, neg_indices):
        new_dirty_df = pd.DataFrame(columns=dataset.dirty_df.columns, dtype=str)
        new_clean_df = pd.DataFrame(columns=dataset.clean_df.columns, dtype=str)

        indices = self._filter(dataset, col, pos_indices, neg_indices)

        # filtered_dataset = dataset

        pos_values = [dataset.dirty_df[col][index] for index in pos_indices] 

        for index, row in dataset.dirty_df.iterrows():
            count = 1
            if index in neg_indices or index in indices:
                   continue
            
            for rule, _ in self.rule2prob.items():
                if rule.validate(row[col]):
                    new_row = row.copy()
                    result = rule.transform(row[col])
                    if result not in pos_values:
                        new_row[col] = result
                        count += 1
                        new_dirty_df = new_dirty_df.append(new_row, ignore_index=True)
                        new_clean_df = new_clean_df.append(
                            row, ignore_index=True
                        )

            new_dirty_df = new_dirty_df.append([row] * count, ignore_index=True)
            new_clean_df = new_clean_df.append(
                [row] * count, ignore_index=True
            )
        return Dataset(new_dirty_df.reset_index(drop=True).sort_index().sort_index(axis=1), new_clean_df.reset_index(drop=True).sort_index().sort_index(axis=1))


class WordNoisyChannel(CharNoisyChannel):
    def learn_transformation(self, error_str, cleaned_str):
        if not cleaned_str and not error_str:
            return []

        error_tokens = wskeep_tokenize(error_str)
        cleaned_tokens = wskeep_tokenize(cleaned_str)

        return self.learn_transformation_tokens(error_tokens, cleaned_tokens)

    def learn_transformation_tokens(self, error_tokens, cleaned_tokens):
        if not [x for x in error_tokens if x] or not [x for x in cleaned_tokens if x]:
            return []

        valid_trans = [WordTransform("".join(cleaned_tokens), "".join(error_tokens))]

        l = self.longest_common_substring(cleaned_tokens, error_tokens)

        if l is None:
            return valid_trans

        lcv, rcv = cleaned_tokens[: l[0]], cleaned_tokens[l[0] + l[2] :]
        lev, rev = error_tokens[: l[1]], error_tokens[l[1] + l[2] :]

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) >= self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or lev:
                valid_trans.append(WordTransform("".join(lcv), "".join(lev)))
            if rcv or rev:
                valid_trans.append(WordTransform("".join(rcv), "".join(rev)))
            valid_trans.extend(self.learn_transformation_tokens(lev, lcv))
            valid_trans.extend(self.learn_transformation_tokens(rev, rcv))

        elif self.similarity(lcv, lev) + self.similarity(rcv, rev) < self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or rev:
                valid_trans.append(WordTransform("".join(lcv), "".join(rev)))
            if rcv or lev:
                valid_trans.append(WordTransform("".join(rcv), "".join(lev)))
            valid_trans.extend(self.learn_transformation_tokens(rev, lcv))
            valid_trans.extend(self.learn_transformation_tokens(lev, rcv))

        return list(set(valid_trans))


class NegNCGenerator:
    def __init__(self, active_learner):
        self.word_channel = WordNoisyChannel()
        self.char_channel = CharNoisyChannel()

        self.active_learner = active_learner

    def _check_exceptions(self, str1, exceptions):
        for exception in exceptions:
            if exception in str1:
                return exception
        return False

    def fit_transform(self, dataset: Dataset, col: str, pos_indices, neg_indices):
        ec_pairs = [
            (dataset.dirty_df[col][i], dataset.clean_df[col][i]) for i in neg_indices
        ]
        start_time = time.time()
        self.char_channel.fit(ec_pairs)
        dataset1 = self.char_channel.transform(dataset, col, pos_indices, neg_indices)
        print(f"Char channel {time.time() - start_time}")

        start_time = time.time()
        self.word_channel.fit(ec_pairs)
        dataset2 = self.word_channel.transform(dataset, col, pos_indices, neg_indices)
        print(f"Word channel {time.time() - start_time}")

        return dataset1 + dataset2


class PosNCGenerator(NegNCGenerator):
    def fit_transform(self, dataset: Dataset, col: str, pos_indices, neg_indices):
        values = dataset.dirty_df[col].values.tolist()
        product_values = list(
            itertools.product(
                random.choices(values, k=100), random.choices(values, k=100),
            )
        )

        self.char_channel.fit(product_values)

        dataset1 = self.char_channel.transform(dataset, col, pos_indices, neg_indices)

        self.word_channel.fit(product_values)
        dataset2 = self.word_channel.transform(dataset, col, pos_indices, neg_indices)
        return dataset1 + dataset2


class CombinedNCGenerator(NegNCGenerator):
    def __init__(self, active_learner):
        self.neg_generator = NegNCGenerator(active_learner)
        self.pos_generator = PosNCGenerator(active_learner)

    def fit_transform(self, dataset: Dataset, col: str, pos_indices, neg_indices):
        dataset1 = self.neg_generator.fit_transform(
            dataset, col, pos_indices, neg_indices
        )

        # dataset2 = self.pos_generator.fit_transform(
        #     dataset, col, pos_indices, neg_indices
        # )
        return dataset1
