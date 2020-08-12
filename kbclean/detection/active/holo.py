import itertools
from logging import exception
import random
from pprint import pprint
from typing import Counter, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import Trainer
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText

from kbclean.detection.base import BaseDetector, BaseModule
from kbclean.detection.extras.highway import Highway
from kbclean.transformation.noisy_channel import NoisyChannel, TransformationRule
from kbclean.utils.data.helpers import split_train_test_dls, unzip_and_stack_tensors
from kbclean.utils.features.holodetect import (
    sym_trigrams,
    sym_value_freq,
    val_trigrams,
    value_freq,
)
from kbclean.utils.logger import MetricsTensorBoardLogger


class HoloExtractor:
    def __init__(self, feature_funcs=None):
        if feature_funcs is None:
            self.feature_funcs = [
                val_trigrams,
                sym_trigrams,
                value_freq,
                sym_value_freq,
            ]
        else:
            self.feature_funcs = feature_funcs

    def extract_features(self, values):
        feature_lists = []
        for func in self.feature_funcs:
            feature_lists.append(func(values))

        feature_vecs = list(zip(*feature_lists))
        return np.asmatrix(feature_vecs)


class LearnableModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.highway = Highway(
            self.hparams.emb_dim, self.hparams.num_layers, torch.nn.functional.relu
        )
        self.linear = nn.Linear(self.hparams.emb_dim, 1)

    def forward(self, inputs):
        avg_input = torch.mean(inputs, dim=1)
        hw_out = self.highway(avg_input)
        return self.linear(hw_out)


class HoloDetector(BaseModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.char_model = LearnableModule(hparams)
        self.word_model = LearnableModule(hparams)

        self.fcs = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.input_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hparams.input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, word_inputs, char_inputs, other_inputs):
        word_out = self.word_model(word_inputs)
        char_out = self.char_model(char_inputs)

        concat_inputs = torch.cat([word_out, char_out, other_inputs], dim=1).float()

        return self.fcs(concat_inputs)

    def training_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class HoloActiveDetector(BaseDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.model = HoloDetector(hparams.detector)
        self.trans_learner = NoisyChannel()
        self.feature_extractor = HoloExtractor()

        self.tokenizer = get_tokenizer("spacy")
        self.fasttext = FastText()
        self.scaler = MinMaxScaler()

        random.seed(1811)

    def prepare(self, save_path):
        pass

    def extract_features(self, data, labels=None):
        features = self.scaler.fit_transform(
            self.feature_extractor.extract_features(data)
        )
        features = torch.tensor(features)

        word_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor

        char_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(list(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor

        if labels is not None:
            label_data = torch.tensor(labels)

            return word_data, char_data, features, label_data
        return word_data, char_data, features

    def check_exceptions(self, str1, exceptions):
        for exception in exceptions:
            if exception in str1:
                return False
        return True

    def generate_transformed_data(
        self, rule2prob, values: List[str], exception_strs: List[str]
    ):
        examples = []
        wait_time = 0

        exceptions = [x.after_str for x in rule2prob.keys()]

        while len(examples) < len(values) and wait_time < len(values):
            val = random.choice(values)

            probs = []
            rules = []

            for rule, prob in rule2prob.items():
                if rule.validate(val) and self.check_exceptions(val, exceptions):
                    rules.append(rule)
                    probs.append(prob)
            if probs:
                rule = random.choices(rules, weights=probs, k=1)[0]
                transformed_value = rule.transform(val[:])
                if val not in exception_strs:
                    examples.append(transformed_value)
                else:
                    wait_time += 1
            else:
                wait_time += 1
            if wait_time == len(values) - 1:
                if exceptions:
                    exceptions.remove(min(exceptions, key=lambda x: len(x)))
                    wait_time = 0
        return examples, exceptions

    def idetect_values(
        self, ec_str_pairs: str, values: List[str], test_values: List[str]
    ):
        rule2prob = self.trans_learner.transformation_distribution(ec_str_pairs)
        logger.info("Rule Probabilities: " + str(rule2prob))

        neg_values, exceptions = self.generate_transformed_data(
            rule2prob, values, [x[1] for x in ec_str_pairs]
        )
        logger.info("Values: " + str(values[:10]))
        logger.info("Exceptions: " + str(exceptions))

        pos_values = [
            x
            for x in list([val for val in values if val not in neg_values])
            if self.check_exceptions(x, exceptions)
        ]

        logger.info("Negative values: " + str(neg_values[:10]))
        logger.info("Positive values: " + str(pos_values[:10]))

        data, labels = (
            neg_values + pos_values,
            [0 for _ in range(len(neg_values))] + [1 for _ in range(len(pos_values))],
        )

        word_data, char_data, features, label_data = self.extract_features(data, labels)

        dataset = TensorDataset(word_data, char_data, features, label_data)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.detector.batch_size,
        )

        trainer = Trainer(
            gpus=4,
            distributed_backend="dp",
            logger=MetricsTensorBoardLogger("tt_logs", "active"),
            max_epochs=20,
        )
        trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=[val_dataloader],
        )

        trainer.save_checkpoint(f"{self.hparams.save_path}/model.ckpt")

        word_data, char_data, features = self.extract_features(test_values)

        pred = self.model.forward(word_data, char_data, features)

        return (pred >= 0.5).squeeze(1).detach().cpu().numpy()

    def detect_values(self, values: List[str]):
        pass

    def detect(self, df: pd.DataFrame):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            outliers = self.detect_values(values)
            result_df[column] = pd.Series(outliers)
        return result_df

    def fake_idetect(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame):
        result_df = raw_df.copy()
        for column in raw_df.columns:
            values = raw_df[column].values.tolist()
            cleaned_values = cleaned_df[column].values.tolist()
            true_values = []
            false_values = []
            for val, cleaned_val in zip(values, cleaned_values):
                if val != cleaned_val:
                    false_values.append((val, cleaned_val))
                else:
                    true_values.append(val)
            if not false_values:
                result_df[column] = pd.Series([True for _ in range(len(raw_df))])
            else:
                outliers = self.idetect_values(false_values[:10], values, values)
                result_df[column] = pd.Series(outliers)
        return result_df
