import functools
import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from kbclean.utils.data.readers import RowBasedValue

import nltk
import numpy as np
import pandas as pd
from kbclean.utils.data.helpers import str2regex
from kbclean.utils.features.attribute import xngrams
from kbclean.utils.search.query import ESQuery
from loguru import logger
from metal.label_model import LabelModel
from metal.end_model import EndModel



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
    def fit(self, row_values):
        pass

    @abstractmethod
    def predict(self, row_values, scores=None):
        pass

    def fit_predict(self, row_values, scores=None):
        self.fit(row_values)

        return self.predict(row_values, scores)

    def reverse_predict(row_values, scores=None):
        pass


class MinValue(BestValuePicker):
    def __init__(self, es_query, func):
        self.es_query = es_query
        self.func = func
        self.value2count = {}

    def call_func(self, raw_df):
        return raw_df.applymap(lambda x: self.func(x))

    def fit(self, row_values):
        str_values = [val.value for val in row_values]
        self.value2count = {
            value: count
            for value, count in zip(str_values, self.func(self.es_query, str_values))
        }

    def predict(self, row_values, scores=None):
        counts = list(map(lambda x: self.value2count[x.value], row_values))

        return (
            np.argmin(counts),
            np.min(counts) / np.mean(counts),
        )

    def reverse_predict(self, row_values, scores):
        counts = list(map(lambda x: self.value2count[x.value], row_values))

        return np.argsort(counts)[-int(len(row_values) * 0.5) :]


class MaxProb(BestValuePicker):
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
            value = self.add_end(value.value)
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

    def predict(self, row_values, scores=None):
        counts = list(map(lambda x: self.sentence_probability(x.value), row_values))

        return (
            np.argmin(counts),
            np.min(counts) / np.mean(counts),
        )

    def reverse_predict(self, row_values, scores):
        counts = list(map(lambda x: self.sentence_probability(x.value), row_values))

        return np.argsort(counts)[-int(len(row_values) * 0.5) :]


class MinCoValue(BestValuePicker):
    def __init__(self):
        self.covalue2count = defaultdict(defaultdict(lambda: 0))
        self.value2count = defaultdict(lambda: 0)

    def fit(self, row_values):
        for row_value in row_values:
            for col_name, col_value in row_value.row_dict.items():
                self.covalue2count[row_value.value][(col_value, col_name)] += 1
                self.value2count[row_value.value] += 1

    def predict(self, row_values, scores=None):
        probs = list(
            map(
                lambda x: min(list(self.covalue2count[x.value].values()))
                / self.value2count[x.value],
                row_values,
            )
        )

        return np.argmin(probs), np.min(probs)

    def reverse_predict(self, row_values, scores=None):
        probs = list(
            map(
                lambda x: min(list(self.covalue2count[x.value].values()))
                / self.value2count[x.value],
                row_values,
            )
        )

        return np.argsort(probs)[-int(len(row_values) * 0.5) :]


class MaxAmbiguous(BestValuePicker):
    def fit(self, row_values):
        pass

    def predict(self, row_values, scores):
        if scores is None:
            return None, float("inf")

        score_arr = np.abs(np.asfarray(scores) - 0.5)
        return np.argmin(score_arr), np.min(score_arr)

    def reverse_predict(self, row_values, scores):
        if scores is None:
            return None, float("inf")

        score_arr = np.abs(np.asfarray(scores) - 0.5)

        return np.argsort(score_arr)[-int(len(score_arr) * 0.5) :]


class ActiveLearner:
    def __init__(self, raw_df, cleaned_df, hparams):
        self.hparams = hparams

        self.raw_df = raw_df
        self.cleaned_df = cleaned_df

        self.raw_records = raw_df.to_dict("records")
        self.cleaned_records = cleaned_df.to_dict("recordss")

        self.raw_col2values = {
            col: [
                RowBasedValue(x, col, row_dict)
                for x, row_dict in zip(
                    self.raw_df[col].values.tolist(), self.raw_records
                )
            ]
            for col in self.raw_df.columns
        }
        self.cleaned_col2values = {
            col: [
                RowBasedValue(x, col, row_dict)
                for x, row_dict in zip(
                    self.cleaned_df[col].values.tolist(), self.cleaned_records
                )
            ]
            for col in self.cleaned_df.columns
        }

        self.es_query = ESQuery.get_instance(hparams.es_host, hparams.es_port)

        self.col2examples = defaultdict(list)

        self.col2criteria = {
            col: {
                "min_ngram": MinValue(self.es_query, min_ngram_counts),
                "min_tok_ngram": MinValue(self.es_query, min_tok_ngram_counts),
                "min_sym_ngram": MinValue(self.es_query, min_sym_ngram_counts),
                "markov_char": MaxProb(),
                "markov_sym": MaxProb(
                    functools.partial(str2regex, match_whole_token=False)
                ) 
            }
            for col in self.raw_col2values.keys()
            # "max_ambiguous": MaxAmbiguous(),
        }

        self.col2chosen_criteria = defaultdict(list)
        self.col2positive_criteria = defaultdict(list)

    def most_positives(self, row_values, scores):
        positive_lists = []
        column_name = row_values[0].column_name

        if not self.col2chosen_criteria[column_name]:
            return row_values


        for name in self.col2chosen_criteria[column_name]:
            positive_lists.append(
                set(self.col2criteria[column_name][name].reverse_predict(row_values, scores))
            )

        final_indices = positive_lists[0]

        for positive_list in positive_lists[1:]:
            final_indices = final_indices.union(positive_list)

        return [row_values[idx] for idx in final_indices]

    def next(self, i, col, score_df):
        best_col = col
        best_score = float("inf")
        best_index = None
        best_criterion = None

        if not self.raw_col2values[best_col]:
            return best_col, [None, None]

        if len(self.col2chosen_criteria[best_col]) == len(self.col2criteria[best_col]):
            self.col2chosen_criteria[best_col]

        for name, criterion in self.col2criteria[col].items():
            if score_df is not None:
                scores = score_df[best_col]
            else:
                scores = None

            if i == 0:
                index, score = criterion.fit_predict(
                    self.raw_col2values[best_col], scores
                )
            else:
                index, score = criterion.predict(self.raw_col2values[best_col], scores)

            if index is not None:
                logger.debug(
                    f"Criterion with example: {name} '{self.raw_col2values[best_col][index].value}' --- Score: {score}"
                )

            if score < best_score:
                best_score = score
                best_criterion = name
                best_index = index

        logger.debug(
            "Best criterion with example: %s '%s' --- Score: %s"
            % (
                str(best_criterion),
                self.raw_col2values[best_col][best_index],
                best_score,
            )
        )
        if best_index is not None:
            best_raw = self.raw_col2values[best_col][best_index]
            best_cleaned = self.cleaned_col2values[best_col][best_index]

            indices = np.where(
                np.asarray([x.value for x in self.raw_col2values[best_col]])
                == best_raw.value
            )[0].tolist()

            for index in reversed(indices):
                self.raw_col2values[best_col].pop(index)
                self.cleaned_col2values[best_col].pop(index)

            if best_raw.value == best_cleaned.value:
                self.col2positive_criteria[best_col].append(best_criterion)
            self.col2chosen_criteria[best_col].append(best_criterion)

            return best_col, (best_raw, best_cleaned)
        return best_col, (None, None)

    def update(self, i, col, scores):
        for j in range(self.hparams.num_examples):
            result = self.next(i * self.hparams.num_examples + j, col, scores)
            if result[1][0] is not None:
                col, (raw_value, cleaned_value) = result
                logger.debug(
                    f'New example in column [{col}]: "{raw_value}" vs "{cleaned_value}"'
                )
                self.col2examples[col].append((raw_value, cleaned_value))
            else:
                logger.debug("No outlier detected in this iteration")



class MetalLeaner(ActiveLearner):
    def __init__(self):
        self.label_model = LabelModel()
        self.end_model = EndModel()

    def next(self, i, col, score_df):
        best_col = col
        best_score = float("inf")
        best_index = None
        best_criterion = None

        if not self.raw_col2values[best_col]:
            return best_col, [None, None]

        if len(self.col2chosen_criteria[best_col]) == len(self.col2criteria[best_col]):
            self.col2chosen_criteria[best_col]

        for name, criterion in self.col2criteria[col].items():
            if score_df is not None:
                scores = score_df[best_col]
            else:
                scores = None

            if i == 0:
                index, score = criterion.fit_predict(
                    self.raw_col2values[best_col], scores
                )
            else:
                index, score = criterion.predict(self.raw_col2values[best_col], scores)

            if index is not None:
                logger.debug(
                    f"Criterion with example: {name} '{self.raw_col2values[best_col][index].value}' --- Score: {score}"
                )

            if score < best_score:
                best_score = score
                best_criterion = name
                best_index = index

        logger.debug(
            "Best criterion with example: %s '%s' --- Score: %s"
            % (
                str(best_criterion),
                self.raw_col2values[best_col][best_index],
                best_score,
            )
        )
        if best_index is not None:
            best_raw = self.raw_col2values[best_col][best_index]
            best_cleaned = self.cleaned_col2values[best_col][best_index]

            indices = np.where(
                np.asarray([x.value for x in self.raw_col2values[best_col]])
                == best_raw.value
            )[0].tolist()

            for index in reversed(indices):
                self.raw_col2values[best_col].pop(index)
                self.cleaned_col2values[best_col].pop(index)

            if best_raw.value == best_cleaned.value:
                self.col2positive_criteria[best_col].append(best_criterion)
            self.col2chosen_criteria[best_col].append(best_criterion)

            return best_col, (best_raw, best_cleaned)
        return best_col, (None, None)