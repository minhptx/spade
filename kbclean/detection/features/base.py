import abc
from abc import abstractmethod
import itertools
from typing import List
from kbclean.utils.data.readers import RowBasedValue


class BaseExtractor(metaclass=abc.ABCMeta):
    @abstractmethod
    def fit(self, values: List[RowBasedValue]):
        pass

    @abstractmethod
    def transform(self, values: List[RowBasedValue]):
        pass

    def fit_transform(self, values: List[RowBasedValue]):
        self.fit(values)

        return self.transform(values)

    @abstractmethod
    def n_features(self):
        pass


class ConcatExtractor(BaseExtractor):
    def __init__(self, name2extractor):
        self.name2extractor = name2extractor

    def fit(self, values: List[RowBasedValue]):
        for extractor in self.name2extractor.values():
            extractor.fit(values)

    def transform(self, values: List[RowBasedValue]):
        return list(itertools.chain.from_iterable([extractor.transform(values) for extractor in self.name2extractor.values()]))

    def __getitem__(self, key):
        return self.name2extractor[key]

    def n_features(self):
        return sum([x.n_features() for x in self.name2extractor.values()])