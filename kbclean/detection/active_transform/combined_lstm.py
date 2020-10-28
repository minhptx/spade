import itertools
from kbclean.detection.features.statistics import StatsExtractor
import os
import random
from functools import partial
from typing import Counter, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from kbclean.detection.active_transform.holo import HoloDetector
from kbclean.detection.active_transform.lstm import Attention
from kbclean.detection.base import ActiveDetector, BaseModule
from kbclean.transformation.noisy_channel import NCGenerator
from kbclean.utils.data.helpers import (
    build_vocab,
    split_train_test_dls,
    str2regex,
    unzip_and_stack_tensors,
)
from pytorch_lightning import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText




class ClapLSTMModel(BaseModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.regex_embedding = nn.Embedding(
            self.hparams.regex_size, self.hparams.reg_emb_dim
        )

        self.char_lstm = nn.GRU(
            300, self.hparams.char_hid_dim, batch_first=True, bidirectional=True
        )

        self.reg_lstm = nn.GRU(
            self.hparams.reg_emb_dim,
            self.hparams.reg_hid_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.char_attention = Attention(self.hparams.char_hid_dim * 2)

        self.reg_attention = Attention(self.hparams.reg_hid_dim * 2)

        self.fc = nn.Linear(
            2 * (self.hparams.char_hid_dim + self.hparams.reg_hid_dim)
            + self.hparams.feature_dim,
            1,
        )

    def forward(self, char_inputs, regex_inputs, lengths, features):
        packed_inputs = pack_padded_sequence(
            char_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.char_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        char_attn_outputs = self.char_attention(outputs)

        regex_inputs = self.regex_embedding(regex_inputs)

        packed_inputs = pack_padded_sequence(
            regex_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.reg_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        reg_attn_outputs = self.reg_attention(outputs)

        attn_outputs = torch.cat([char_attn_outputs, reg_attn_outputs, features], dim=1)

        return torch.sigmoid(self.fc(attn_outputs))

    def training_step(self, batch, batch_idx):
        word_inputs, regex_inputs, lengths, features, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, regex_inputs, lengths, features)

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        word_inputs, regex_inputs, lengths, features, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, regex_inputs, lengths, features)

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class ClapLSTMDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = StatsExtractor()

        self.generator = NCGenerator()

        self.fasttext = FastText()

    def extract_features(self, data, labels=None, retrain=False):
        regex_data = [str2regex(x, match_whole_token=False) for x in data]

        if labels:
            features = self.feature_extractor.fit_transform(data)
            self.hparams.model.feature_dim = self.feature_extractor.n_features()
        else:
            features = self.feature_extractor.transform(data)

        if retrain:
            self.regex_vocab = build_vocab(
                list(itertools.chain.from_iterable([regex_data]))
            )
            self.hparams.model.regex_size = len(self.regex_vocab)
            self.model = ClapLSTMModel(self.hparams.model)

        char_data, lengths = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(list(str_value.lower()))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        )

        regex_data, _ = stack_and_pad_tensors(
            [
                torch.tensor(self.regex_vocab.lookup_indices(list(str_value)))
                if str_value
                else torch.zeros(1).long()
                for str_value in regex_data
            ]
        )

        if labels is not None:
            label_data = torch.tensor(labels)

            return char_data, regex_data, lengths, torch.tensor(features), label_data
        return char_data, regex_data, lengths, torch.tensor(features)

    def reset(self):
        self.model = ClapLSTMModel(self.hparams.model)

    def idetect_values(self, ec_str_pairs: str, values: List[str], recommender):
        print("Values", values[:10])
        data, labels = self.generator.fit_transform(ec_str_pairs, values, recommender)

        feature_tensors_with_labels = self.extract_features(data, labels, retrain=True)

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
                early_stop_callback=None,
                val_percent_check=0,
                checkpoint_callback=False,
                logger=False
            )

            trainer.fit(
                self.model, train_dataloader=train_dataloader,
            )

        feature_tensors = self.extract_features(values, retrain=False)

        self.model.eval()
        pred = self.model.forward(*feature_tensors)

        return pred.squeeze().detach().cpu().numpy()

