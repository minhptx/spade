import itertools
import math
import os
import random
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kbclean.detection.active.holo import HoloDetector
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
        self.char_regex_lstm = nn.LSTM(
            self.hparams.char_emb_dim, self.hparams.hid_dim, batch_first=True
        )
        self.attention = Attention(
            self.hparams.hid_dim, self.hparams.hid_dim, self.hparams.hid_dim
        )
        self.fc = nn.Linear(self.hparams.hid_dim, 1)

    def forward(self, char_inputs, regex_inputs, lengths):
        regex_embed = self.regex_embedding(regex_inputs)
        char_embed = self.char_embedding(char_inputs)
        reg_char_inputs = char_embed
        # reg_char_inputs = torch.cat([char_embed, regex_embed], dim=2)
        packed_inputs = pack_padded_sequence(
            reg_char_inputs, lengths, batch_first=True, enforce_sorted=False
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


class OCLSTMModel(BaseModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.regex_embedding = nn.Embedding(
            self.hparams.regex_size, self.hparams.reg_emb_dim
        )
        self.char_embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.char_emb_dim
        )
        self.char_regex_lstm = nn.LSTM(
            self.hparams.char_emb_dim, self.hparams.hid_dim, batch_first=True
        )
        self.attention = Attention(
            self.hparams.hid_dim, self.hparams.hid_dim, self.hparams.hid_dim
        )
        self.fc = nn.Linear(self.hparams.hid_dim, self.hparams.hid_dim // 2)

    def forward(self, char_inputs, regex_inputs, lengths):
        regex_embed = self.regex_embedding(regex_inputs)
        char_embed = self.char_embedding(char_inputs)
        # reg_char_inputs = char_embed
        reg_char_inputs = torch.cat([char_embed, regex_embed], dim=2)
        packed_inputs = pack_padded_sequence(
            reg_char_inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h, c) = self.char_regex_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        attn_outputs = self.attention(h, outputs, outputs)
        return attn_outputs, self.fc(attn_outputs)

    def nnscore(x, w, v):
        return torch.matmul(torch.matmul(x, w), v)

    def ocnn_loss(self, y_pred, r):
        term1 = 0.5 * torch.sum(w1 ** 2)
        term2 = 0.5 * torch.sum(w2 ** 2)
        term3 = 1 / self.hparams.nu * torch.mean(F.relu(r - y_pred))
        term4 = -r

        return term1 + term2 + term3 + term4

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


class LSTMDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.generator = NCGenerator()

        if hparams.gen_method == "DSL":
            logger.warning("Running DSL")
            self.generator = DSLGenerator()

    def extract_features(self, data, labels=None):
        data = ["^" + x for x in data]
        regex_data = [str2regex(x, match_whole_token=False) for x in data]
        if labels is not None:
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
                if str_value
                else torch.zeros(1, 300)
                for str_value in regex_data
            ]
        )

        if labels is not None:
            label_data = torch.tensor(labels)

            return char_data, regex_data, lengths, label_data
        return char_data, regex_data, lengths

    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        data, labels = self.generator.fit_transform(ec_str_pairs, values)
        feature_tensors_with_labels = self.extract_features(data, labels)

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.model.batch_size,
        )

        self.model = LSTMModel(self.hparams.model)
        self.model.train()

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

        trainer.save_checkpoint(f"{self.hparams.save_path}/model.ckpt")

        feature_tensors = self.extract_features(values)

        self.model.eval()
        pred = self.model.forward(*feature_tensors)
        return pred.squeeze().detach().cpu().numpy()
