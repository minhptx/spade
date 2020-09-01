import pandas as pd
from nltk.util import ngrams

from kbclean.utils.es import ESQuery


class CandidateGeneration:
    def __init__(self, hparams):

