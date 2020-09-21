from collections import defaultdict
from loguru import logger
import numpy as np
import pandas as pd
from kbclean.utils.features.attribute import xngrams
from kbclean.utils.search.query import ESQuery
from kbclean.utils.data.helpers import str2regex


class Uncommoner:
    def __init__(self, raw_df, cleaned_df, hparams):
        self.raw_df = raw_df
        self.cleaned_df = cleaned_df
        self.es_query = ESQuery.get_instance(hparams.es_host, hparams.es_port)

        self.col2examples = defaultdict(list)

        self.raw_col2values = {col: self.raw_df[col].values.tolist() for col in self.raw_df.columns}
        self.cleaned_col2values = {col: self.cleaned_df[col].values.tolist() for col in self.cleaned_df.columns}

    def min_ngram_counts(self, values):
        return [
            min(
                self.es_query.get_char_ngram_counts(
                    ["".join(x) for x in xngrams(list(value), 3, False)]
                )
            )
            for value in values
        ]

    def min_tok_ngram_counts(self, values):
        return [
            min(
                self.es_query.get_char_ngram_counts(
                    ["".join(x) for x in xngrams(list(value), 3, False)]
                )
            )
            for value in values
        ]

    def min_coexist(self, values):
        coexist_count = self.es_query.get_coexist_counts(values)
        return pd.DataFrame(coexist_count).to_numpy()

    def next_column(self):
        return self.raw_df.columns[0]

    def best_feature(self, col):
        best_sum_distance = -1
        best_func = None
        for func in [self.col_min_ngrams, self.col_min_sym_ngrams]:
            counts = func(col)
            func_sum_distance = np.mean(np.abs(counts - np.mean(counts)))
            if func_sum_distance > best_sum_distance and np.min(counts) < 1000:
                best_sum_distance = func_sum_distance
                best_func = func
        return best_func, best_sum_distance

    def col_min_ngrams(self, column):
        min_ngram_counts = self.min_ngram_counts(self.raw_col2values[column])
        return min_ngram_counts

    def col_min_sym_ngrams(self, column):
        sym_values = [str2regex(x, False) for x in self.raw_col2values[column]]
        min_ngram_counts = self.min_ngram_counts(sym_values)
        return min_ngram_counts

    def first_example(self, best_col):
        best_col_distance = -1
        best_col_func = None

        best_func, best_distance = self.best_feature(best_col)
        if best_distance > best_col_distance:
            best_col_func = best_func
            best_col_distance = best_distance

        if best_col is not None:
            counts = best_col_func(best_col)
            index = np.argmin(counts)
            raw_value = self.raw_col2values[best_col].pop(index)
            cleaned_value = self.cleaned_col2values[best_col].pop(index)

            indices = np.where(self.raw_col2values[best_col] == raw_value)[0]

            for index in reversed(indices):
                self.raw_col2values[best_col].pop(index)
                self.cleaned_col2values[best_col].pop(index)

            if raw_value != cleaned_value:
                return raw_value, cleaned_value
        
        return None

    def most_ambigous(self, best_col, scores_df):
        best_col_df = scores_df[scores_df["col"] == best_col]
        best_col_df["am_score"] = best_col_df["score"].apply(lambda x: abs(x - 0.5))
        max_index = best_col_df["am_score"].argmin()

        return best_col_df["from"][max_index].item(), best_col_df["to"][max_index].item()

    def next(self, i, scores_df):
        best_col = self.next_column()
        if i == 0:
            return best_col, self.first_example(best_col)
        else:
            return best_col, self.most_ambigous(best_col, scores_df)
        

    def update(self, i, scores):
        result = self.next(i, scores)
        if result[1] is not None:
            col, (raw_value, cleaned_value) = result
            logger.debug(f'New example in column [{col}]: "{raw_value}" vs "{cleaned_value}"')
            self.col2examples[col].append({"raw": raw_value, "cleaned": cleaned_value})
        else:
            logger.debug("No outlier detected in this iteration")
        return self.col2examples