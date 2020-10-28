from datetime import datetime
import functools
import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import lru_cache
from datasketch.lsh import MinHashLSH
import nltk

import numpy as np
import pandas as pd
from kbclean.utils.data.helpers import str2regex
from kbclean.utils.features.attribute import xngrams
from kbclean.utils.search.query import ESQuery
from loguru import logger
from datasketch import MinHash


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


def min_sym_ngram_counts(es_query, values):
    sym_values = [str2regex(x, False) for x in values]
    trigrams = list(
        itertools.chain.from_iterable(
            [
                ["".join(x) for x in xngrams(list(value), 3, False)]
                for value in sym_values
            ]
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
        for value in sym_values
    ]


def min_tok_ngram_counts(es_query, values):
    trigrams = list(
        itertools.chain.from_iterable(
            [
                ["".join(x) for x in xngrams(nltk.wordpunct_tokenize(value), 3, False)]
                for value in values
            ]
        )
    )
    counts = es_query.get_tok_ngram_counts(trigrams)
    value2count = {trigram: count for trigram, count in zip(trigrams, counts)}

    return [
        min(
            [
                value2count[x]
                for x in [
                    "".join(x)
                    for x in xngrams(nltk.wordpunct_tokenize(value), 3, False)
                ]
            ]
        )
        for value in values
    ]


class BestValuePicker(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, values):
        pass

    @abstractmethod
    def predict(self, values):
        pass

    def fit_predict(self, values):
        self.fit(values)

        return self.predict(values)

    def reverse_predict(values):
        pass


class Uncommoner(BestValuePicker):
    def __init__(self, es_query, func):
        self.es_query = es_query
        self.func = func
        self.value2count = {}

    def call_func(self, value):
        return self.func(value)

    def fit(self, values):
        self.value2count = {
            value: count
            for value, count in zip(values, self.func(self.es_query, values))
        }

    def predict(self, values):
        counts = [self.value2count[value] for value in values]
        return np.argmin(counts), np.min(counts) / np.max(counts)

    def reverse_predict(self, values):
        counts = [self.value2count[value] for value in values]
        sorted_indices = np.argsort(counts)[::-1]

        return sorted_indices[: int(len(values) * 0.2)]


class MarkovModel(BestValuePicker):
    def __init__(self, func=None):
        self.bigram2prob = defaultdict(lambda: 0)
        self.unigram2prob = defaultdict(lambda: 0)
        self.func = func

    def add_end(self, value):
        if self.func is None:
            value = list(value) + ["<END>"]
        else:
            value = list(self.func(value)) + ["<END>"]
        return value

    def fit(self, values):
        for value in values:
            value = self.add_end(value)
            for i in range(len(value) - 1):
                self.bigram2prob[(value[i], value[i + 1])] += 1
                self.unigram2prob[value[i]] += 1

            self.unigram2prob[value[-1]] += 1

    def sentence_probability(self, value):
        value = self.add_end(value)
        prob = 1

        for i in range(len(value) - 1):
            prob *= (
                self.bigram2prob[(value[i], value[i + 1])]
                * 1.0
                / self.unigram2prob[value[i]]
            )

        return prob

    def predict(self, values):
        probs = [self.sentence_probability(value) for value in values]
        return np.argmin(probs), np.min(probs)

    def reverse_predict(self, values):
        probs = [self.sentence_probability(value) for value in values]
        sorted_indices = np.argsort(probs)[::-1]

        return sorted_indices[: int(len(values) * 0.2)]


class ActiveLearner:
    def __init__(self, raw_df, cleaned_df, hparams):
        self.hparams = hparams

        self.raw_df = raw_df
        self.cleaned_df = cleaned_df
        self.es_query = ESQuery.get_instance(hparams.es_host, hparams.es_port)

        self.col2examples = defaultdict(list)

        self.raw_col2values = {
            col: self.raw_df[col].values.tolist() for col in self.raw_df.columns
        }
        self.cleaned_col2values = {
            col: self.cleaned_df[col].values.tolist() for col in self.cleaned_df.columns
        }

        self.name2criteria = {
            "min_ngram": Uncommoner(self.es_query, min_ngram_counts),
            "min_tok_ngram": Uncommoner(self.es_query, min_tok_ngram_counts),
            "min_sym_ngram": Uncommoner(self.es_query, min_sym_ngram_counts),
            "markov_char": MarkovModel(),
            "markov_sym": MarkovModel(
                functools.partial(str2regex, match_whole_token=False)
            ),
            "markov_tok_sym": MarkovModel(str2regex),
        }

        self.chosen_criteria = []
        self.positive_criteria = []

    def next_column(self):
        return self.raw_df.columns[0]

    def most_ambigous(self, best_col, scores_df):
        best_col_df = scores_df[scores_df["col"] == best_col]
        best_col_df["am_score"] = best_col_df["score"].apply(lambda x: abs(x - 0.5))
        max_index = best_col_df["am_score"].argmin()

        return (
            best_col_df["from"][max_index].item(),
            best_col_df["to"][max_index].item(),
        )

    def most_positives(self, values):
        positive_lists = []
        if not self.positive_criteria:
            return values
        for name in self.positive_criteria:
            positive_lists.append(set(self.name2criteria[name].reverse_predict(values)))

        final_indices = positive_lists[0]

        for positive_list in positive_lists[1:]:
            final_indices = final_indices.union(positive_list)

        return [values[index] for index in final_indices]

    def next(self, i, scores_df):
        best_col = self.next_column()
        best_score = 10
        best_example = None
        best_cleaned_example = None
        best_criterion = None

        if not self.raw_col2values[best_col]:
            return best_col, [None, None]

        if len(self.chosen_criteria) == len(self.name2criteria):
            self.chosen_criteria = []

        for name, criterion in self.name2criteria.items():
            if name in self.chosen_criteria:
                continue
            if i == 0:
                index, score = criterion.fit_predict(self.raw_col2values[best_col])
            else:
                index, score = criterion.predict(self.raw_col2values[best_col])

            logger.debug(
                f"Criterion with example: {name} '{self.raw_col2values[best_col][index]}' --- Score: {score}"
            )

            if score < best_score:
                best_score = score
                best_criterion = name
                best_example = self.raw_col2values[best_col][index]
                best_cleaned_example = self.cleaned_col2values[best_col][index]

        logger.debug(
            "Best criterion with example: %s '%s' --- Score: %s"
            % (str(best_criterion), best_example, best_score)
        )
        if best_example is not None:
            indices = np.where(np.array(self.raw_col2values[best_col]) == best_example)[
                0
            ].tolist()

            for index in reversed(indices):
                self.raw_col2values[best_col].pop(index)
                self.cleaned_col2values[best_col].pop(index)

            if best_example != best_cleaned_example:
                self.positive_criteria.append(best_criterion)
            self.chosen_criteria.append(best_criterion)

            return best_col, (best_example, best_cleaned_example)
        else:
            if i == 0:
                return best_col, (None, None)
            else:
                example = self.most_ambigous(best_col, scores_df)
                indices = np.where(
                    np.array(self.raw_col2values[best_col]) == example[0]
                )[0].tolist()
                for index in reversed(indices):
                    self.raw_col2values[best_col].pop(index)
                    self.cleaned_col2values[best_col].pop(index)

                return best_col, example

    def update(self, i, scores):
        for j in range(self.hparams.num_examples):
            result = self.next(i * self.hparams.num_examples + j, scores)
            if result[1][0] is not None:
                col, (raw_value, cleaned_value) = result
                logger.debug(
                    f'New example in column [{col}]: "{raw_value}" vs "{cleaned_value}"'
                )
                self.col2examples[col].append(
                    {"raw": raw_value, "cleaned": cleaned_value}
                )
            else:
                logger.debug("No outlier detected in this iteration")
        return self.col2examples


class MinimalPairFinder:
    def __init__(self):
        self.lsh = MinHashLSH(threshold=0.8, num_perm=256)

    def min_hash(self, value):
        min_hash = MinHash()
        for c in set(list(value)):
            min_hash.update(c.encode("utf8"))

        return min_hash

    def fit(self, values):
        for value in values:
            self.lsh.insert(value, self.min_hash(value))

    def transform(self, values):
        pass
