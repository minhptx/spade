import math
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kbclean.detection.base import ActiveDetector, BaseModule
from kbclean.utils.data.helpers import (split_train_test_dls,
                                        unzip_and_stack_tensors)
from kbclean.utils.logger import MetricsTensorBoardLogger
from pytorch_lightning import Trainer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import euclidean_distances
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data import get_tokenizer
from torchtext.experimental.vectors import FastText


class Encoder(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, nlayers=1, dropout=0.0, bidirectional=True
    ):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            nlayers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, inputs, lengths, hidden=None):
        packed = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, hidden = self.lstm(packed, hidden)
        return pad_packed_sequence(packed_output, batch_first=True)[0], hidden


class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1.0 / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[BxT], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1, 2)  # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(
            1
        )  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class Distancer(nn.Module):
    def __init__(self, encoder, attention, hparams):
        super(Distancer, self).__init__()
        self.encoder = encoder
        self.attention = attention
        if hparams.bidirection:
            self.decoder = nn.Linear(hparams.hid_dim * 2, hparams.out_dim)
        else:
            self.decoder = nn.Linear(hparams.hid_dim, hparams.out_dim)

    def encode(self, inputs, lengths):
        outputs, hidden = self.encoder(inputs, lengths)
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[1]  # take the cell state

        if self.encoder.bidirectional:  # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        energy, linear_combination = self.attention(hidden, outputs, outputs)
        logits = self.decoder(linear_combination)
        return logits, energy

    def forward(self, inputs1, lengths1, inputs2, lengths2):
        encoded1, _ = self.encode(inputs1, lengths1)
        encoded2, _ = self.encode(inputs2, lengths2)
        return torch.exp(
            -torch.abs(torch.cdist(encoded1.unsqueeze(1), encoded2.unsqueeze(1), 1))
        ).squeeze()


class DistanceMaximizer(BaseModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.encoder = Encoder(
            hparams.emb_dim,
            hparams.hid_dim,
            hparams.n_layers,
            hparams.dropout,
            hparams.bidirection,
        )
        self.attention = Attention(hparams.hid_dim)
        self.model = Distancer(self.encoder, self.attention, hparams)

    def encode(self, inputs, lengths):
        return self.model.encode(inputs, lengths)[0].detach().cpu().numpy()

    def forward(self, inputs1, lengths1, inputs2, lengths2):
        return self.model.forward(inputs1, lengths1, inputs2, lengths2)

    def training_step(self, batch, batch_idx):
        inputs1, lengths1, inputs2, lengths2, labels = batch

        probs = self.forward(inputs1, lengths1, inputs2, lengths2)
        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (preds == labels).sum().float() / preds.shape[0]

        logs = {"train_loss": loss, "train_acc": acc}

        return {
            "loss": loss,
            "acc": acc,
            "log": logs,
            "progress_bar": logs,
        }

    def validation_step(self, batch, batch_idx):
        inputs1, lengths1, inputs2, lengths2, labels = batch

        probs = self.forward(inputs1, lengths1, inputs2, lengths2)

        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (preds == labels).sum().float() / preds.shape[0]

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class DistanceDetector(ActiveDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.model = DistanceMaximizer(hparams.maximizer)
        self.tokenizer = get_tokenizer("spacy")
        self.fasttext = FastText()

    def generate_training_data(self, data_with_labels):
        org_strs = [x[0] for x in data_with_labels]
        trg_strs = [x[1] for x in data_with_labels]
        labels = [x[2] for x in data_with_labels]

        error_seqs, error_lengths = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in org_strs
            ]
        )

        cleaned_seqs, cleaned_lengths = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in trg_strs
            ]
        )

        return (
            error_seqs,
            error_lengths,
            cleaned_seqs,
            cleaned_lengths,
            torch.tensor(labels),
        )

    def idetect_values(
        self, ec_str_pairs: List, values: List[str], test_values: List[str]
    ):

        str_pairs = [
            (str_pair[0], clean_str, 0)
            for clean_str in values
            for str_pair in ec_str_pairs
            if str_pair[0] != clean_str
        ] + [(str1, str2, 1) for str1 in values for str2 in values]
        train_tensors = self.generate_training_data(str_pairs)

        dataset = TensorDataset(*train_tensors)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.detector.batch_size,
        )

        trainer = Trainer(
            gpus=0,
            distributed_backend="dp",
            logger=MetricsTensorBoardLogger("tt_logs", "active"),
            max_epochs=10,
        )
        trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=[val_dataloader],
        )

        trainer.save_checkpoint(f"{self.hparams.save_path}/model.ckpt")

        test_tensors = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in test_values
            ]
        )

        encoded_test_data = self.model.encode(*test_tensors)

        dist_df = pd.DataFrame(
            euclidean_distances(encoded_test_data, encoded_test_data),
            columns=test_values,
            index=test_values,
        )

        isolation_forest = IsolationForest()
        outliers = isolation_forest.fit_predict(encoded_test_data)
        return outliers != -1
