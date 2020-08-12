import pandas as pd
from nltk.util import ngrams

from kbclean.utils.es import ESQuery


class CandidateGeneration:
    def __init__(self, hparams):
        self.es_query: ESQuery = ESQuery.get_instance(hparams.es_host, hparams.es_port)

    def generate_candidates(self, values):
        min_ngram_counts = self.min_ngram_counts(values)
        coexist_matrix = self.min_coexist(values)
        cand_indices  = []

        for idx, count in enumerate(min_ngram_counts):
            if count < 20:
                cand_indices.append(idx)

        for idx in range(len(values)):
            if min(coexist_matrix[idx]) < 10:
                cand_indices.append(idx)
            elif min_ngram_counts[idx] < 10:
                cand_indices.append(idx)

    def min_ngram_counts(self, values):
        return self.es_query.get_char_ngram_counts(values)

    def min_coexist(self, values):
        coexist_count = self.es_query.get_coexist_counts(values)
        return pd.DataFrame(coexist_count).to_numpy()
