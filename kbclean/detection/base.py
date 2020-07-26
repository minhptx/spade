from abc import ABCMeta, abstractmethod
from sys import modules
from typing import List

import pandas as pd
import torch
from pytorch_lightning import LightningModule


class BaseDetector(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare(self, save_path):
        pass

    @abstractmethod
    def detect(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @staticmethod
    def load(load_path, hparams):
        pass


class BaseModule(LightningModule):
    def on_train_start(self):
        self.logger.log_hyperparams_metrics(
            self.hparams,
            {"val_loss": 100, "val_acc": 0, "train_loss": 100, "train_acc": 0},
        )

    def training_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        logs = {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
        }
        return {"avg_train_loss": avg_loss, "log": logs, "progress_bar": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"avg_val_loss": avg_loss, "log": logs, "progress_bar": logs}


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
