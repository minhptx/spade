import time
from loguru import logger
from pytorch_lightning.core.lightning import LightningModule
from kbclean.detection.features.base import ConcatExtractor
from kbclean.detection.features.embedding import (
    CharFastText,
    CoValueFastText,
    WordFastText,
)
from kbclean.utils.data.readers import RowBasedValue
import os
import random
from functools import partial
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from kbclean.detection.active_transform.holo import HoloDetector
from kbclean.detection.features.statistics import StatsExtractor
from kbclean.transformation.noisy_channel import CombinedNCGenerator, NCGenerator, SameNCGenerator
from kbclean.utils.data.helpers import (
    split_train_test_dls,
    unzip_and_stack_tensors,
)
from pytorch_lightning import Trainer
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import TensorDataset
from imblearn.over_sampling import SMOTE


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)

        self.W = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(1))
        self.feature_dim = feature_dim

    def forward(self, x):
        eij = (
            torch.mm(x.view(-1, self.feature_dim), self.W).view(-1, x.shape[1]) + self.b
        )

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, dim=1)


class LSTMModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.char_lstm = nn.GRU(
            300, self.hparams.char_hid_dim, batch_first=True, bidirectional=True
        )

        self.word_lstm = nn.GRU(
            300, self.hparams.word_hid_dim, batch_first=True, bidirectional=True
        )

        self.char_attention = Attention(self.hparams.char_hid_dim * 2)

        self.word_attention = Attention(self.hparams.word_hid_dim * 2)

        self.fc = nn.Linear(
            2 * (self.hparams.char_hid_dim + self.hparams.word_hid_dim) + self.hparams.feature_dim, 1,
        )

    def forward(self, char_inputs, char_lengths, word_inputs, word_lengths, features):
        packed_inputs = pack_padded_sequence(
            char_inputs, char_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.char_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        char_attn_outputs = self.char_attention(outputs)

        packed_inputs = pack_padded_sequence(
            word_inputs, word_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.word_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        word_attn_outputs = self.word_attention(outputs)

        attn_outputs = torch.cat([char_attn_outputs, word_attn_outputs, features], dim=1)

        return torch.sigmoid(self.fc(attn_outputs))

    def training_step(self, batch, batch_idx):
        char_inputs, char_lengths, word_inputs, word_length, features, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, char_lengths, word_inputs, word_length, features)

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        char_inputs, char_lengths, word_inputs, word_length, features, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, char_lengths, word_inputs, word_length, features)

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", loss, on_epoch=True)

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class LSTMDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = ConcatExtractor(
            {
                "char_ft": CharFastText(),
                "word_ft": WordFastText(),
                "stats": StatsExtractor(),
                # "covalue_ft": CoValueFastText(),
            }
        )

    def extract_features(self, data, labels=None):
        if labels:
            features = self.feature_extractor.fit_transform(data)
            self.hparams.model.feature_dim = self.feature_extractor[
                "stats"
            ].n_features()
        else:
            features = self.feature_extractor.transform(data)

        if labels is not None:
            label_data = torch.tensor(labels)

            return features + [label_data]
        return features

    def reset(self):
        self.model = LSTMModel(self.hparams.model)

    def idetect_values(
        self, ec_str_pairs: str, row_values: List[RowBasedValue], scores, recommender
    ):
        generator = CombinedNCGenerator(recommender)
        start_time = time.time()  
        data, labels = generator.fit_transform(ec_str_pairs, row_values, scores)

        logger.info(f"Total transformation time: {time.time() - start_time}")

        feature_tensors_with_labels = self.extract_features(data, labels)


        self.model = LSTMModel(self.hparams.model)

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, _, _ = split_train_test_dls(
            dataset,
            unzip_and_stack_tensors,
            self.hparams.model.batch_size,
            ratios=[1, 0],
            num_workers=16,
            pin_memory=False,
        )

        self.model.train()

        if len(train_dataloader) > 0:
            os.environ["MASTER_PORT"] = str(random.randint(49152, 65535))

            trainer = Trainer(
                gpus=self.hparams.num_gpus,
                distributed_backend="dp",
                max_epochs=self.hparams.model.num_epochs,
                checkpoint_callback=False,
                logger=False,
            )

            trainer.fit(
                self.model, train_dataloader=train_dataloader,
            )

        feature_tensors = self.extract_features(row_values)

        self.model.eval()
        pred = self.model.forward(*feature_tensors)

        return pred.squeeze().detach().cpu().numpy()

    def idetect(self, df: pd.DataFrame, score_df: pd.DataFrame, recommender):
        result_df = df.copy()
        records = df.to_dict("records")
        for column in df.columns:
            values = df[column].values.tolist()
            row_values = [
                RowBasedValue(value, column, row_dict)
                for value, row_dict in zip(values, records)
            ]
            if score_df is not None:
                scores = score_df[column].values.tolist()
            else:
                scores = None
            if (
                column not in recommender.col2examples
                or not recommender.col2examples[column]
                or all(
                    [x[0].value == x[1].value for x in recommender.col2examples[column]]
                )
            ):
                result_df[column] = pd.Series([1.0 for _ in range(len(df))])
            else:
                outliers = self.idetect_values(
                    recommender.col2examples[column], row_values, scores, recommender,
                )
                result_df[column] = pd.Series(outliers)
        return result_df
