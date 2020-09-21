import abc


class Recommender(metaclass=abc.ABCMeta):
    def recommend(self, chosen_indices=None, predictions=None, previous_policies=None):