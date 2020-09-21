import itertools
from logging import exception
from kbclean.utils.features.attribute import xngrams
import random
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Tuple

from loguru import logger


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

    def __eq__(self, o: "TransformationRule"):
        return self.before_str == o.before_str and self.after_str == o.after_str

    def __hash__(self) -> int:
        return hash(f"Rule('{self.before_str}', '{self.after_str}')")

    def validate(self, str_value):
        return self.before_str in str_value

    def __repr__(self) -> str:
        return f"Rule('{self.before_str}', '{self.after_str}')"


class NoisyChannel:
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

        if self.similarity(lcv, lev) + self.similarity(rcv, rev) <= self.similarity(
            lcv, rev
        ) + self.similarity(rcv, lev):
            if lcv or rev:
                valid_trans.append(TransformationRule(lcv, rev))
            if rcv or lev:
                valid_trans.append(TransformationRule(rcv, lev))
            valid_trans.extend(self.learn_transformation(rev, lcv))
            valid_trans.extend(self.learn_transformation(lev, rcv))

        return list(set(valid_trans))

    def fit(self, string_pairs):
        transforms = []
        for error_str, cleaned_str in string_pairs:
            transforms.extend(self.learn_transformation(error_str, cleaned_str))

        logger.debug(transforms)
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


class NCGenerator:
    def __init__(self):
        self.trans_learner = NoisyChannel()

    def _check_exceptions(self, str1, exceptions):
        for exception in exceptions:
            if exception in str1:
                return exception
        return False

    # def _generate_transformed_data(self, values: List[str], cleaned_strs: List[str]):
    #     examples = []
    #     wait_time = 0

    #     exceptions = [
    #         x.after_str for x in self.trans_learner.rule2prob.keys() if x.after_str not in cleaned_strs and x.after_str and x.before_str
    #     ]

    #     exceptions.extend(itertools.chain.from_iterable([list(x.after_str) for x in self.trans_learner.rule2prob.keys() if not x.before_str]))

    #     print(exceptions)

    #     exception_hits = []

    #     while len(examples) < len(values) and wait_time <= len(values):
    #         val = random.choice(values)

    #         probs = []
    #         rules = []
    #         for rule, prob in self.trans_learner.rule2prob.items():
    #             check_result = self._check_exceptions(val, exceptions)

    #             if rule.validate(val):
    #                 if check_result == False:
    #                     rules.append(rule)
    #                     probs.append(prob)
    #                 else:
    #                     exception_hits.append(check_result)
    #         if probs:
    #             rule = random.choices(rules, weights=probs, k=1)[0]
    #             transformed_value = rule.transform(val[:])
    #             examples.append(transformed_value)
    #         else:
    #             wait_time += 1
    #         if wait_time == len(values) and exceptions and exception_hits:
    #             exception_mode = Counter(exception_hits).most_common(1)[0][0]
    #             exception_hits = []
    #             exceptions.remove(exception_mode)
    #             wait_time = 0
    #     return examples, exceptions

    def _generate_transformed_data(self, values: List[str]):
        examples = []
        wait_time = 0

        while len(examples) < len(values) and wait_time <= len(values):
            val = random.choice(values)

            probs = []
            rules = []

            for rule, prob in self.trans_learner.rule2prob.items():
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

    # def fit_transform(self, ec_pairs: List[Tuple[str, str]], values: List[str]):
    #     logger.debug(f"Pair examples: {ec_pairs}")
    #     self.trans_learner.fit(ec_pairs)
    #     logger.debug("Rule Probabilities: " + str(self.trans_learner.rule2prob))

    #     neg_values, exceptions = self._generate_transformed_data(
    #         values, [x[1] for x in ec_pairs if x[0] != x[1]]
    #     )

    #     logger.debug("Values: " + str(set(values)))
    #     logger.debug("Exceptions: " + str(exceptions))

    #     abs_pos_values = [x[0] for x in ec_pairs if x[0] == x[1]]

    #     pos_values = [
    #         x
    #         for x in list([val for val in values if val not in neg_values])
    #         if x not in abs_pos_values and not self._check_exceptions(x, exceptions)
    #     ]

    #     logger.debug(
    #         f"{len(neg_values)} negative values: " + str(list(neg_values)[:20])
    #     )
    #     logger.debug(
    #         f"{len(pos_values)} positive values: " + str(list(pos_values)[:20])
    #     )

    #     data, labels = (
    #         neg_values + abs_pos_values + pos_values,
    #         [0 for _ in range(len(neg_values))] + [1 for _ in range(len(abs_pos_values))] + [0.7 for _ in range(len(pos_values))],
    #     )

    #     return data, labels


    def fit_transform(self, ec_pairs: List[Tuple[str, str]], values: List[str]):
        logger.debug(f"Pair examples: {ec_pairs}")
        
        self.trans_learner.fit([x for x in ec_pairs if x[0] != x[1]])
        logger.debug("Rule Probabilities: " + str(self.trans_learner.rule2prob))

        neg_values = self._generate_transformed_data(
            values
        ) + [x[0] for x in ec_pairs]

        logger.debug("Values: " + str(set(values)))

        pos_values = [
            x
            for x in list([val for val in values if val not in neg_values])
        ] + [x[0] for x in ec_pairs if x[0] == x[1]]

        logger.debug(
            f"{len(neg_values)} negative values: " + str(list(neg_values)[:20])
        )
        logger.debug(
            f"{len(pos_values)} positive values: " + str(list(pos_values)[:20])
        )

        data, labels = (
            neg_values + pos_values,
            [0 for _ in range(len(neg_values))] + [1 for _ in range(len(pos_values))],
        )

        return data, labels
