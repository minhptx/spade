import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from kbclean.detection.base import BaseModule


class TransClassifier(BaseModule):
    def __init__(self, hparams) -> None:
        self.hparams = hparams

        self.rule_embedding = nn.Embedding(
            self.hparams.rule_vocab_size, self.hparams.rule_emb_dim
        )
        self.input_embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.emb_dim
        )
        self.output_embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.emb_dim
        )

        self.input_lstm = nn.LSTM(
            self.hparams.emb_dim, self.hparams.hid_dim, batch_first=True
        )
        self.output_lstm = nn.LSTM(
            self.hparams.emb_dim, self.hparams.hid_dim, batch_first=True
        )

        self.linear = nn.Sequential(
            nn.Linear(self.hparams.hid_dim, self.hparams.linear_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.linear_dim, self.hparams.linear_dim),
            nn.Sigmoid(),
        )

    def forward(self, rules, inputs, input_lengths, outputs, output_lengths):
        r_embed = self.rule_embedding(rules)
        i_embed = self.input_embedding(inputs)
        o_embed = self.output_embedding(outputs)

        r_embed_seq = r_embed.unsqueeze(1).repeat(i_embed.shape[1])
        i_combined_embed = torch.cat([r_embed_seq, i_embed], dim=2)

        packed_i = pack_padded_sequence(i_combined_embed, input_lengths, batch_first=True)
        _, (h, c) = self.input_lstm(packed_i)

        h_seq = h.unsqueeze(1).repeat(o_embed.shape[1])
        o_combined_embed = torch.cat([h_seq, o_embed], dim=2)

        packed_o = pack_padded_sequence(o_combined_embed, output_lengths, batch_first=True)
        _, (h1, _) = self.output_lstm(packed_o)

        return self.linear(h1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(*x)

        loss = F.binary_cross_entropy(y_hat, y)
        acc = (y == y_hat).sum() * 1.0 / y.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(*x)

        loss = F.binary_cross_entropy(y_hat, y)
        acc = (y == y_hat).sum() * 1.0 / y.shape[0]

        return {"val_loss": loss, "val_acc": acc}
