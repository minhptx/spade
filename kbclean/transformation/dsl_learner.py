import itertools
import string
from collections import defaultdict
from typing import List, Tuple
from loguru import logger
import regex as re
from functools import reduce

token2regex = {
    "digit": r"[0-9]+",
    "lowercase": r"[a-z]+",
    "uppercase": r"[A-Z]+",
    "alphabet": r"[A-Za-z]+",
    "alphanum": r"[A-Za-z0-9]+",
    "whitespace": r"\s+",
}

token2regex.update({x: re.escape(x) for x in string.punctuation})


class Token:
    def __init__(self, token_type):
        self.type = token_type


class DAG:
    def __init__(self, interval2transform, length):
        self.interval2transform = interval2transform
        self.length = length

    def __call__(self, raw_str):
        transforms = [
            [[] for _ in range(self.length + 1)] for _ in range(self.length + 1)
        ]
        for i in self.interval2transform.keys():
            for j in self.interval2transform[i].keys():
                transforms[i][j] = list(
                    set(
                        [
                            x(raw_str)
                            for x in self.interval2transform[i][j]
                            if x(raw_str)
                        ]
                    )
                )

        result = [[] for _ in range(self.length + 1)]
        for i in range(1, self.length + 1):
            result[i] = transforms[0][i]
            for j in range(i):
                result[i].extend(
                    [
                        x[0] + x[1]
                        for x in itertools.product(result[j], transforms[j][i])
                    ]
                )
            result[i] = list(set(result[i]))
        return [x for x in result[self.length] if x]


class ConstStr:
    def __init__(self, value):
        self.value = value

    def __call__(self, raw_str):
        return self.value

    def __repr__(self) -> str:
        return f"ConstStr('{self.value}')"


class Substr:
    def __init__(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos

    def __call__(self, raw_str):
        pos1 = self.start_pos(raw_str)
        pos2 = self.end_pos(raw_str)
        return raw_str[pos1:pos2]

    def __repr__(self) -> str:
        return f"Substr('{self.start_pos}, {self.end_pos}')"


class CPos:
    def __init__(self, pos):
        self.pos = pos

    def __call__(self, raw_str):
        return self.pos

    def __repr__(self) -> str:
        return f"CPos('{self.pos}')"


class RPos:
    def __init__(self, regex, count):
        self.regex = regex
        self.count = count

    def __call__(self, raw_str):
        matches = list(re.finditer(f"{self.regex}", raw_str))
        if self.count < 0:
            count = len(matches) - self.count
        else: 
            count = self.count
        if count >= len(matches) or count < 0:
            return None
        match = matches[self.count]
        return match.start()

    def __repr__(self) -> str:
        return f"RPos('{self.regex}, {self.count}')"


class DSLLearner:
    def __init__(self):
        pass

    def fit(self, str_pairs):
        return [self.generate_str(input_str, output_str) for input_str, output_str in str_pairs]

    def get_iparts(self, input_str):
        token2matches = defaultdict(list)
        for token, regex in token2regex.items():
            token2matches[token] = re.findall(regex, input_str)
        iparts = list(token2matches.keys())
        remove_list = []
        for i in range(len(iparts)):
            token1 = iparts[i]
            for j in range(i, len(iparts)):
                token2 = iparts[j]
                if (
                    token1 != token2
                    and token2matches[token1] == token2matches[token2]
                    and token2 in iparts
                ):
                    remove_list.append(token2)
        return [x for x in iparts if x not in remove_list]

    def generate_str(self, input_str, output_str):
        interval2transform = defaultdict(lambda: defaultdict(list))
        for i in range(len(output_str)):
            for j in range(i + 1, len(output_str) + 1):
                substr_ops = self.generate_substr(input_str, output_str[i:j])
                if substr_ops:
                    interval2transform[i][j].extend(substr_ops)
                else:
                    interval2transform[i][j].append(ConstStr(output_str[i:j]))
        return DAG(interval2transform, len(output_str))

    def generate_substr(self, input_str, output_str):
        result = []
        iparts = self.get_iparts(output_str)
        for match in re.finditer(output_str, input_str):
            for pos1 in self.generate_pos(input_str, match.start(), iparts):
                for pos2 in self.generate_pos(input_str, match.end(), iparts):
                    result.append(Substr(pos1, pos2))
        return result

    def generate_pos(self, input_str, k, iparts):
        result = [CPos(k), CPos(-(len(input_str) - k))]
        for token1 in iparts:
            regex1 = token2regex[token1]
            if re.match(regex1, input_str[k - 1]):
                length = 0
                c = -1
                matches = re.finditer(f"{regex1}", input_str)
                for match in matches:
                    if match.start() == k:
                        c = length
                    length += 1
                if c != -1:
                    result.append(RPos(regex1, c))
                    result.append(RPos(regex1, -(length - c)))
        return result


class DSLGenerator:
    def __init__(self):
        self.trans_learner = DSLLearner()

    
    def _check_exceptions(self, str1, exceptions):
        for exception in exceptions:
            if exception in str1:
                return exception
        return False

    def _get_exceptions(self, dags):
        exceptions = []
        for dag in dags:
            for i in dag.interval2transform.keys():
                for trans_funcs in dag.interval2transform[i].values():
                    for trans_func in trans_funcs:
                        if isinstance(trans_func, ConstStr):
                            exceptions.append(trans_func.value)

        return exceptions


    def _generate_transformed_data(
        self, dags, values: List[str], exceptions: List[str]
    ):
        result_lists = []
        for dag in dags:
            result_lists.append(list(itertools.chain.from_iterable([dag(value) for value in values])))
        intersection = reduce(set.intersection, [set(l) for l in result_lists])


        return [x for x in intersection if not self._check_exceptions(x, exceptions)]

    def fit_transform(self, ec_pairs: List[Tuple[str, str]], values: List[str]):
        dags = self.trans_learner.fit(ec_pairs)

        exceptions = self._get_exceptions(dags)

        neg_values = self._generate_transformed_data(
            dags, values, exceptions
        )

        logger.debug("Values: " + str(set(values)))

        pos_values = [
            x
            for x in list([val for val in values if val not in neg_values])
            if not self._check_exceptions(x, exceptions)
        ]

        logger.debug(
            f"{len(neg_values)} negative values: " + str(list(set(neg_values))[:20])
        )
        logger.debug(
            f"{len(pos_values)} positive values: " + str(list(set(pos_values))[:20])
        )

        data, labels = (
            neg_values + pos_values,
            [0 for _ in range(len(neg_values))] + [1 for _ in range(len(pos_values))],
        )

        return data, labels