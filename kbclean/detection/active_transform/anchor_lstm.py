import itertools
import math
import os
import random
from typing import List
from unicodedata import bidirectional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kbclean.detection.active_transform.holo import HoloDetector
from kbclean.detection.active_transform.lstm import Attention
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
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch.autograd.variable import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText


class AnchorLSTMModel(BaseModule):
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

        self.lstm = nn.LSTM(
            self.hparams.reg_emb_dim,
            self.hparams.reg_hid_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = Attention(
            self.hparams.hid_dim,
            self.hparams.id_dim,
            self.hparams.hid_dim,
        )

        self.fc = nn.Linear(self.hparams.char_hid_dim + self.hparams.reg_hid_dim, 1)

    def encode(self, char_inputs, regex_inputs, lengths):
        regex_embed = self.regex_embedding(regex_inputs)
        char_embed = self.char_embedding(char_inputs)
        reg_char_inputs = Variable(torch.cat([char_embed, regex_embed], dim=2), requires_grad=True)

        reg_char_inputs = self.dropout(reg_char_inputs)

        packed_inputs = pack_padded_sequence(
            reg_char_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h, c) = self.regex_lstm(packed_inputs)
        h = h[0:1, :, :] + h[1:, :, :]

        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        regex_attn_outputs = self.regex_attention(h, outputs, outputs)

        packed_inputs = pack_padded_sequence(
            char_embed, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h, c) = self.word_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = h[0:1, :, :] + h[1:, :, :]

        char_attn_outputs = self.char_attention(h, outputs, outputs)

        attn_outputs = torch.cat([regex_attn_outputs, char_attn_outputs], dim=1)

        return torch.sigmoid(self.fc(attn_outputs))

    def forward(self, char_inputs, regex_inputs, lengths, anchor_char_inputs, anchor_regex):
        pass
        

    def training_step(self, batch, batch_idx):
        word_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, regex_inputs, lengths)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        word_inputs, regex_inputs, lengths, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, regex_inputs, lengths)
        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []
