from abc import ABCMeta, abstractmethod


class BaseActiveLearner(metaclass=ABCMeta):
    def __init__(self, df):
        self.df = df

    @abstractmethod
    def next(self):
        pass



