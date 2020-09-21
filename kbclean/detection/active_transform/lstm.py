import itertools
import math
import os
import random
from typing import List

import numpy as np
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kbclean.detection.active_transform.holo import HoloDetector
from kbclean.detection.base import BaseModule
from kbclean.transformation.dsl_learner import DSLGenerator
from kbclean.transformation.noisy_channel import NCGenerator
from kbclean.utils.data.helpers import (
    build_vocab,
    split_train_test_dls,
    str2regex,
    unzip_and_stack_tensors,
)
from loguru import logger
from numpy.lib.twodim_base import tri
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1.0 / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = energy:[BxT], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.transpose(0, 1)  # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1, 2)  # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(
            1
        )  # [Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination


class LSTMModel(BaseModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.regex_embedding = nn.Embedding(
            self.hparams.regex_size, self.hparams.reg_emb_dim
        )
        self.char_embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.char_emb_dim
        )

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.char_regex_lstm = nn.LSTM(
            self.hparams.reg_emb_dim,
            self.hparams.hid_dim,
            batch_first=True,
        )
        self.attention = Attention(
            self.hparams.hid_dim, self.hparams.hid_dim, self.hparams.hid_dim
        )
        self.fc = nn.Linear(self.hparams.hid_dim, 1)

    def forward(self, char_inputs, regex_inputs, lengths):
        regex_embed = self.regex_embedding(regex_inputs)
        # reg_char_inputs = char_embed
        # reg_char_inputs = Variable(torch.cat([char_embed, regex_embed], dim=2), requires_grad=True)
        reg_inputs = Variable(regex_embed, requires_grad=True)
        

        reg_inputs = self.dropout(reg_inputs)

        packed_inputs = pack_padded_sequence(
            reg_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h, c) = self.char_regex_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        attn_outputs = self.attention(h, outputs, outputs)
        return torch.sigmoid(self.fc(attn_outputs))

    def training_step(self, batch, batch_idx):
        char_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, regex_inputs, lengths)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        char_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(char_inputs, regex_inputs, lengths)
        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class CoTeachingLSTMModel(LightningModule):
    def __init__(self, hparams, char_vocab):
        super().__init__()
        self.hparams = hparams
        
        self.first_model = LSTMModel(hparams)
        self.second_model = LSTMModel(hparams)

        self.char_vocab = char_vocab

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
        char_inputs, regex_inputs, lengths, labels = batch
        
        if optimizer_idx == 0:
            y1 = self.first_model(char_inputs, regex_inputs, lengths)
            y2 = self.second_model(char_inputs, regex_inputs, lengths)
            self.labels = labels.view(-1, 1).float()

            preds_1 = y1 >= 0.5
            self.acc_1 = (self.labels == preds_1).sum().float() / labels.shape[0]

            preds_2 = y2 >= 0.5
            self.acc_2 = (self.labels == preds_2).sum().float() / labels.shape[0]

            self.loss_1, self.loss_2, indices_1, indices_2 = self.loss_coteaching(
                y1, y2, self.labels, self.rate_schedule[self.current_epoch]
            )
            logger.debug(f"Big-loss examples 1: {[(''.join(self.char_vocab.lookup_tokens(char_inputs[index].cpu().numpy().tolist())).strip(), labels[index]) for index in indices_1]}")
            logger.debug(f"Big-loss examples 2: {[(''.join(self.char_vocab.lookup_tokens(char_inputs[index].cpu().numpy().tolist())).strip(), labels[index]) for index in indices_2]}")


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
        probs= self.forward(char_inputs, regex_inputs, lengths)
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

        if hparams.gen_method == "DSL":
            logger.warning("Running DSL")
            self.generator = DSLGenerator()


    def extract_features(self, data, labels=None, retrain=False):
        data = ["^" + x for x in data]
        regex_data = [str2regex(x, match_whole_token=False) for x in data]
        if retrain:
            self.char_vocab = build_vocab(list(itertools.chain.from_iterable(data)))
            self.hparams.model.vocab_size = len(self.char_vocab)
            self.regex_vocab = build_vocab(
                list(itertools.chain.from_iterable([regex_data]))
            )
            self.hparams.model.regex_size = len(self.regex_vocab)

        char_data, lengths = stack_and_pad_tensors(
            [
                torch.tensor(self.char_vocab.lookup_indices(list(str_value)))
                for str_value in data
            ]
        )

        regex_data, _ = stack_and_pad_tensors(
            [
                torch.tensor(self.regex_vocab.lookup_indices(list(str_value)))
                for str_value in regex_data
            ]
        )

        if labels is not None:
            label_data = torch.tensor(labels)

            return char_data, regex_data, lengths, label_data
        return char_data, regex_data, lengths

    def reset(self):
        self.model = CoTeachingLSTMModel(self.hparams.model, self.model.char_vocab)

    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        data, labels = self.generator.fit_transform(ec_str_pairs, values)

        if self.model is None:
            feature_tensors_with_labels = self.extract_features(
                data, labels, retrain=True
            )
            self.model = CoTeachingLSTMModel(self.hparams.model, self.char_vocab)

        else:
            feature_tensors_with_labels = self.extract_features(
                data, labels, retrain=False
            )

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, _, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.model.batch_size, ratios=[1, 0, 0]
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
                self.model,
                train_dataloader=train_dataloader,
            )

        feature_tensors = self.extract_features(values)

        # self.model.eval()
        pred = self.model.forward(*feature_tensors)
        return pred.squeeze().detach().cpu().numpy()


class LSTMNaiveDetector(LSTMDetector):
    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        data, labels = (
            [x[0] for x in ec_str_pairs],
            [0 for _ in range(len(ec_str_pairs))],
        )
        data = np.repeat(data, len(values) // len(ec_str_pairs)).tolist()
        labels = np.repeat(labels, len(values) // len(ec_str_pairs)).tolist()
        data.extend(values)
        labels.extend([0.6 for _ in range(len(values))])

        if self.model is None:
            feature_tensors_with_labels = self.extract_features(
                data, labels, retrain=True
            )
            self.model = LSTMModel(self.hparams.model)

        else:
            feature_tensors_with_labels = self.extract_features(
                data, labels, retrain=False
            )

        feature_tensors_with_labels = self.extract_features(data, labels)

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.model.batch_size,
        )

        self.model.train()

        if len(train_dataloader) > 0:
            os.environ["MASTER_PORT"] = str(random.randint(49152, 65535))

            trainer = Trainer(
                gpus=self.hparams.num_gpus,
                distributed_backend="dp",
                # logger=MetricsTensorBoardLogger("tt_logs", name="active"),
                max_epochs=20,
            )

            trainer.fit(
                self.model,
                train_dataloader=train_dataloader,
                val_dataloaders=[val_dataloader],
            )

        feature_tensors = self.extract_features(values)

        self.model.eval()
        pred = self.model.forward(*feature_tensors)
        return pred.squeeze().detach().cpu().numpy()