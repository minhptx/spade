import itertools
import math
import os
import random
from typing import List
from unicodedata import bidirectional

import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.autograd.variable import Variable
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kbclean.detection.active_transform.holo import HoloDetector, HoloLearnableModule
from kbclean.detection.base import BaseModule
from kbclean.transformation.noisy_channel import NCGenerator
from kbclean.utils.data.helpers import (
    build_vocab,
    split_train_test_dls,
    str2regex,
    unzip_and_stack_tensors,
)
from loguru import logger
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)

        self.W = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(1))
        self.feature_dim = feature_dim

    def forward(self, x):
        eij = torch.mm(x.view(-1, self.feature_dim), self.W).view(-1, x.shape[1]) + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, dim=1)


class LSTMModel(nn.Module):
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
            2 * (self.hparams.char_hid_dim + self.hparams.reg_hid_dim), 1
        )

    def forward(self, char_inputs, regex_inputs, lengths):
        packed_inputs = pack_padded_sequence(
            char_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.char_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        char_attn_outputs = self.char_attention(outputs)

        regex_inputs = Variable(self.regex_embedding(regex_inputs), requires_grad=True)

        packed_inputs = pack_padded_sequence(
            regex_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.reg_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        reg_attn_outputs = self.reg_attention(outputs)

        attn_outputs = torch.cat([char_attn_outputs, reg_attn_outputs], dim=1)

        return torch.sigmoid(self.fc(attn_outputs))


class DistLSTMModel(BaseModule):
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

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.fc = nn.Linear(
            2 * (self.hparams.char_hid_dim + self.hparams.reg_hid_dim), 1
        )

    def forward(self, char_inputs, regex_inputs, lengths):
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

        attn_outputs = torch.cat([char_attn_outputs, reg_attn_outputs], dim=1)

        return self.dropout(attn_outputs)

    def training_step(self, batch, batch_idx):
        word_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        outputs = self.forward(word_inputs, regex_inputs, lengths)

        mean_outputs = torch.mean(outputs, dim=0)
        distance = outputs - mean_outputs

        probs = torch.sigmoid(self.fc(torch.cat([outputs], dim=1)))

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        word_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        outputs = self.forward(word_inputs, regex_inputs, lengths)

        mean_outputs = torch.mean(outputs, dim=0)
        distance = outputs - mean_outputs
        probs = torch.sigmoid(self.fc(torch.cat([outputs], dim=1)))
        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class CoTeachingLSTMModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.first_model = LSTMModel(hparams)
        self.second_model = LSTMModel(hparams)

        self.rate_schedule = np.ones(hparams.num_epochs) * hparams.forget_rate
        self.rate_schedule[: hparams.num_gradual] = np.linspace(
            0, hparams.forget_rate ** hparams.exponent, hparams.num_gradual
        )

    def loss_coteaching(self, y_1, y_2, t, forget_rate):
        loss_1 = F.binary_cross_entropy(y_1, t, reduction="none")
        ind_1_sorted = torch.argsort(loss_1.squeeze())

        loss_2 = F.binary_cross_entropy(y_2, t, reduction="none")
        ind_2_sorted = torch.argsort(loss_2.squeeze())

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * loss_1.shape[0])

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.binary_cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.binary_cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return loss_1_update, loss_2_update, ind_1_sorted[-2:], ind_2_sorted[-2:]

    def forward(self, char_inputs, regex_inputs, lengths):
        result1 = self.first_model(char_inputs, regex_inputs, lengths)
        result2 = self.second_model(char_inputs, regex_inputs, lengths)
        return (result1 + result2) / 2

    def training_step(self, batch, batch_idx, optimizer_idx):
        word_inputs, regex_inputs, lengths, labels = batch

        if optimizer_idx == 0:
            y1 = self.first_model(word_inputs, regex_inputs, lengths)
            y2 = self.second_model(word_inputs, regex_inputs, lengths)
            self.labels = labels.view(-1, 1).float()

            preds_1 = y1 >= 0.5
            self.acc_1 = (self.labels == preds_1).sum().float() / labels.shape[0]

            preds_2 = y2 >= 0.5
            self.acc_2 = (self.labels == preds_2).sum().float() / labels.shape[0]

            self.loss_1, self.loss_2, indices_1, indices_2 = self.loss_coteaching(
                y1, y2, self.labels, self.rate_schedule[self.current_epoch]
            )

            logs = {
                "train_loss1": self.loss_1,
                "train_acc1": self.acc_1,
                "train_loss2": self.loss_2,
                "train_acc2": self.acc_2,
            }
            return {
                "loss": self.loss_1,
                "acc": self.acc_1,
                "log": logs,
                "progress_bar": logs,
            }

        if optimizer_idx == 1:
            logs = {
                "train_loss1": self.loss_1,
                "train_acc1": self.acc_1,
                "train_loss2": self.loss_2,
                "train_acc2": self.acc_2,
            }
            return {
                "loss": self.loss_2,
                "acc": self.acc_2,
                "log": logs,
                "progress_bar": logs,
            }

    def validation_step(self, batch, batch_idx):
        char_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, regex_inputs, lengths)
        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return (
            [
                optim.AdamW(self.first_model.parameters(), lr=self.hparams.lr),
                optim.AdamW(self.second_model.parameters(), lr=self.hparams.lr),
            ],
            [],
        )


class LSTMDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.generator = NCGenerator()
        self.model = None

        self.fasttext = FastText()
        self.tokenizer = get_tokenizer("spacy")

    def extract_features(self, data, labels=None, retrain=False):
        regex_data = [str2regex(x, match_whole_token=False) for x in data]

        if retrain:
            self.regex_vocab = build_vocab(
                list(itertools.chain.from_iterable([regex_data]))
            )
            self.hparams.model.regex_size = len(self.regex_vocab)
            self.model = CoTeachingLSTMModel(self.hparams.model)

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

            return char_data, regex_data, lengths, label_data
        return char_data, regex_data, lengths

    def reset(self):
        self.model = CoTeachingLSTMModel(self.hparams.model)

    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        data, labels = self.generator.fit_transform(ec_str_pairs, values)

        feature_tensors_with_labels = self.extract_features(data, labels, retrain=True)

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, _, _ = split_train_test_dls(
            dataset,
            unzip_and_stack_tensors,
            self.hparams.model.batch_size,
            ratios=[1, 0, 0],
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
            )

            trainer.fit(
                self.model, train_dataloader=train_dataloader,
            )

        feature_tensors = self.extract_features(values)

        self.model.eval()
        pred = self.model.forward(*feature_tensors)
        return pred.squeeze().detach().cpu().numpy()


class LSTM2Detector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.generator = NCGenerator()

        self.fasttext = FastText()

    def extract_features(self, data, labels=None, retrain=False):
        regex_data = [str2regex(x, match_whole_token=False) for x in data]

        if retrain:
            self.regex_vocab = build_vocab(
                list(itertools.chain.from_iterable([regex_data]))
            )
            self.hparams.model.regex_size = len(self.regex_vocab)
            self.model = DistLSTMModel(self.hparams.model)

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

            return char_data, regex_data, lengths, label_data
        return char_data, regex_data, lengths

    def reset(self):
        self.model = DistLSTMModel(self.hparams.model)

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
            num_workers=1,
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
            )

            trainer.fit(
                self.model, train_dataloader=train_dataloader,
            )

        feature_tensors = self.extract_features(values, retrain=False)

        self.model.eval()
        outputs = self.model.forward(*feature_tensors)

        mean_outputs = torch.mean(outputs, dim=0)
        distance = outputs - mean_outputs
        pred = torch.sigmoid(self.model.fc(torch.cat([outputs], dim=1)))

        return pred.squeeze().detach().cpu().numpy()

