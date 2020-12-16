from abc import ABCMeta, abstractmethod
from sys import modules
from typing import Dict, List, Tuple

import pandas as pd
import torch
from pytorch_lightning import LightningModule


class BaseDetector(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def is_interactive(self):
        return False

    @abstractmethod
    def detect(self, df: pd.DataFrame):
        pass


class ActiveDetector(BaseDetector, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def is_interactive(self):
        return True

    @abstractmethod
    def detect(self, df: pd.DataFrame):
        raise NotImplementedError(f"{self.__class__.__name__} is an interactive model.")

    
    def detect(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def idetect(self, df: pd.DataFrame, col2examples: Dict[str, List[Tuple[str, str]]]):
        pass

class Module(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, train_path):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @staticmethod
    def load(load_path):
        pass


class Pipeline:
    def __init__(self, modules, hparams):
        self.hparams = hparams
        self.modules: List[Module] = modules

    def fit(self, train_path):
        for module in modules:
            try:
                module.__class__.load(self.hparams.save_path)
            except:
                module.fit(train_path)

    def save(self, save_path):
        for module in modules:
            module.save(save_path)
