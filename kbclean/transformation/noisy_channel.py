from logging import exception
import random
from collections import Counter
from difflib import SequenceMatcher

import regex as re
from numpy.lib.function_base import append


class TransformationRule:
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
                    + str_value[:sample_position]
                )
        return str_value.replace(self.before_str, self.after_str)

    def __eq__(self, o: 'TransformationRule'):
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __hash__(self) -> int:
        return hash(f"Rule('{self.before_str}', '{self.after_str}')")

    def validate(self, str_value):
        return self.before_str in str_value

    def __repr__(self) -> str:
        return f"Rule('{self.before_str}', '{self.after_str}')"


class NoisyChannel:
    def __init__(self):
        pass

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

        valid_trans = [TransformationRule(cleaned_str, error_str)]

        l = self.longest_common_substring(cleaned_str, error_str)

        if l is None:
            return valid_trans

        lcv, rcv = cleaned_str[: l[0]], cleaned_str[l[0] + l[2] :]
        lev, rev = error_str[: l[1]], error_str[l[1] + l[2] :]

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) >= self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or lev:
                valid_trans.append(TransformationRule(lcv, lev))
            if rcv or rev:
                valid_trans.append(TransformationRule(rcv, rev))
            valid_trans.extend(self.learn_transformation(lev, lcv))
            valid_trans.extend(self.learn_transformation(rev, rcv))

        elif self.similarity(lcv, lev) + self.similarity(rcv, rev) <= self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or rev:
                valid_trans.append(TransformationRule(lcv, rev))
            if rcv or lev:
                valid_trans.append(TransformationRule(rcv, lev))
            valid_trans.extend(self.learn_transformation(rev, lcv))
            valid_trans.extend(self.learn_transformation(lev, rcv))

        return list(set(valid_trans))

    def transformation_distribution(self, string_pairs):
        transforms = []
        for error_str, cleaned_str in string_pairs:
            transforms.extend(self.learn_transformation(error_str, cleaned_str))

        counter = Counter(transforms)
        sum_counter = sum(counter.values())
        return {
            transform: count * 1.0 / sum_counter for transform, count in counter.items()
        }

    def get_noisy_policy(self, str1, transform_dist):
        noisy_policy = {}

        for (error_str, cleaned_str), prob in transform_dist.items():
            if str1 in error_str:
                noisy_policy[(error_str, cleaned_str)] = prob

        sum_prob = sum(noisy_policy.values)
        noisy_policy = list(map(lambda x: (x[0], x[1] / sum_prob)))
        return noisy_policy
