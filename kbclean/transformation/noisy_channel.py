import functools
import itertools
from kbclean import recommendation
import random
from collections import Counter
from difflib import SequenceMatcher
from functools import partial
from logging import exception
from typing import List, Tuple

import regex as re
from kbclean.utils.features.attribute import xngrams
from loguru import logger


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

        logger.debug("Transform rules: " + str(transforms))
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


class WordNoisyChannel(CharNoisyChannel):
    def learn_transformation(self, error_str, cleaned_str):
        if not cleaned_str and not error_str:
            return []

        error_tokens = wskeep_tokenize(error_str)
        cleaned_tokens = wskeep_tokenize(cleaned_str)

        return self.learn_transformation_tokens(error_tokens, cleaned_tokens)

    def learn_transformation_tokens(self, error_tokens, cleaned_tokens):
        if not error_tokens and not cleaned_tokens:
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


class NCGenerator:
    def __init__(self):
        self.word_channel = WordNoisyChannel()
        self.char_channel = CharNoisyChannel()

    def _check_exceptions(self, str1, exceptions):
        for exception in exceptions:
            if exception in str1:
                return exception
        return False

    def _get_suspicious_chars(self, channel):
        for rule in channel.rule2prob.keys():
            if not rule.before_str:
                yield rule.after_str

    def _generate_transformed_data(self, channel, values: List[str]):

        examples = []
        wait_time = 0

        while len(examples) < len(values) and wait_time <= len(values):
            val = random.choice(values)

            probs = []
            rules = []

            for rule, prob in channel.rule2prob.items():
                if rule.validate(val):
                    rules.append(rule)
                    probs.append(prob)

            if probs:
                rule = random.choices(rules, weights=probs, k=1)[0]
                transformed_value = rule.transform(val[:])
                examples.append(transformed_value)
            else:
                wait_time += 1

        return examples

    def _filter_normal_values(self, channel, values):
        suspicious_chars = self._get_suspicious_chars(channel)
        all_remove_values = []
        for c in suspicious_chars:
            removed_values = []
            for value in values:
                if c in value:
                    removed_values.append(value)
            if len(removed_values) < len(values) * 0.2:
                all_remove_values.extend(removed_values)

        for value in set(all_remove_values):
            values.remove(value)

        logger.debug("Remove_values", all_remove_values)
        return values

    def fit_transform_channel(
        self, channel, ec_pairs: List[Tuple[str, str]], values: List[str], recommender
    ):
        logger.debug(f"Pair examples: {ec_pairs}")

        neg_pairs = [x for x in ec_pairs if x[0] != x[1]]
        pos_pairs = [x for x in ec_pairs if x[0] == x[1]]

        values = recommender.most_positives(values)

        channel.fit(neg_pairs)
        logger.debug("Rule Probabilities: " + str(channel.rule2prob))
        neg_values = [x[0] for x in neg_pairs] + self._generate_transformed_data(
            channel, values
        )

        logger.debug("Values: " + str(set(values)))

        pos_values = [x for x in list([val for val in values if val not in neg_values])]

        pos_values = self._filter_normal_values(channel, pos_values) + [
            x[0] for x in pos_pairs
        ]

        logger.debug(
            f"{len(neg_values)} negative values: " + str(list(set(neg_values)))
        )
        logger.debug(
            f"{len(pos_values)} positive values: " + str(list(set(pos_values)))
        )

        data, labels = (
            neg_values + pos_values,
            [0 for _ in range(len(neg_values))] + [1 for _ in range(len(pos_values))],
        )

        return data, labels

    def fit_transform(self, ec_pairs: List[Tuple[str, str]], values: List[str], recommender):
        data1, labels1 = self.fit_transform_channel(self.char_channel, ec_pairs, values, recommender)
        data2, labels2 = self.fit_transform_channel(self.word_channel, ec_pairs, values, recommender)
        return data1 + data2, labels1 + labels2
