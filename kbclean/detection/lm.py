from collections import OrderedDict
import itertools
from typing import Counter

from torch import optim
from kbclean.utils.logger import MetricsTensorBoardLogger
from pytorch_lightning.trainer.trainer import Trainer
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.experimental.vectors import FastText
from torchtext.experimental.vocab import Vocab
from kbclean.detection.base import BaseDetector, BaseModule
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from kbclean.utils.data.helpers import split_train_test_dls, unzip_and_stack_tensors
import torch.nn.functional as F


class RNNModel(BaseModule):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, hparams):
        super(RNNModel, self).__init__()
        self.hparams = hparams
        self.drop = nn.Dropout(hparams.dropout)
        self.rnn = nn.LSTM(
            hparams.emb_dim,
            hparams.hid_dim,
            hparams.n_layers,
            dropout=hparams.dropout,
            batch_first=True,
        )
        self.decoder = nn.Linear(hparams.hid_dim, hparams.vocab_size)
    
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(input)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return torch.softmax(decoded, dim=2)

    def predict_proba(self, embed_inputs, word_inputs):
        probs = self.forward(embed_inputs, self.init_hidden(embed_inputs))
        output_probs = torch.gather(probs, dim=2, index=word_inputs.unsqueeze(2))
        return torch.mean(output_probs.log(), dim=1)

    def init_hidden(self, batch_ex):
        return (
            torch.zeros(
                self.hparams.n_layers, batch_ex.shape[0], self.hparams.hid_dim
            ).type_as(batch_ex),
            torch.zeros(
                self.hparams.n_layers, batch_ex.shape[0], self.hparams.hid_dim
            ).type_as(batch_ex),
        )

    def training_step(self, batch, batch_idx):
        embed_data, word_data = batch
        probs = self.forward(embed_data[:, :-1], self.init_hidden(embed_data))
        loss = F.cross_entropy(
            probs.view(-1, self.hparams.vocab_size), word_data[:, 1:].reshape(-1)
        )
        preds = torch.argmax(probs, dim=2).view(-1, 1)
        acc = (preds == word_data[1:].view(-1)).sum().float() / preds.shape[0]

        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        embed_data, word_data = batch
        probs = self.forward(embed_data[:, :-1], self.init_hidden(embed_data))
        loss = F.cross_entropy(
            probs.view(-1, self.hparams.vocab_size), word_data[:, 1:].reshape(-1)
        )
        preds = torch.argmax(probs, dim=2).view(-1, 1)
        acc = (preds == word_data[1:].view(-1)).sum().float() / preds.shape[0]

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class LMDetector(BaseDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.tokenizer = get_tokenizer("spacy")
        self.fasttext = FastText()

    def prepare(self, save_path):
        pass

    def extract_features(self, data, build_vocab=False):
        if build_vocab:
            tokenized_data = [["<eos>"] + self.tokenizer(val) for val in data]
            all_tokens = itertools.chain.from_iterable(tokenized_data)
            counter = Counter(all_tokens)
            sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            self.vocab = Vocab(OrderedDict(sorted_counter))

        embed_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(["<s>"] + self.tokenizer(str_value))
                for str_value in data
            ]
        ).tensor

        word_data = stack_and_pad_tensors(
            [
                torch.tensor(self.vocab.lookup_indices(["<s>"] + self.tokenizer(str_value)))
                for str_value in data
            ]
        ).tensor

        return embed_data, word_data

    def detect_values(self, values):
        embed_data, word_data = self.extract_features(values, build_vocab=True)

        self.hparams.model.vocab_size = len(self.vocab)

        self.model = RNNModel(self.hparams.model)

        dataset = TensorDataset(embed_data, word_data)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset, unzip_and_stack_tensors, self.hparams.model.batch_size,
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

        embed_data, word_data = self.extract_features(values)
        preds = []
        probs = []
        for i in range(embed_data.shape[0]):
            pred = (
                self.model.predict_proba(embed_data[i : i + 1], word_data[i : i + 1])
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )
            preds.append((pred >= 0.5)[0])
            probs.append(pred)
        return preds, probs

    def detect(self, df):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            outliers, probs = self.detect_values(values)
            result_df[column] = pd.Series(outliers)
        return result_df

