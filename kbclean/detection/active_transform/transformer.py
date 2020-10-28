import itertools

from torchtext.experimental.vectors import FastText
from kbclean.utils.data.dataset.huggingface import HuggingfaceDataset

from torch import optim
from kbclean.detection.base import BaseModule
import math
import os
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from kbclean.detection.active_transform.holo import HoloDetector
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
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class TransformerModel(BaseModule):
    def __init__(self, hparams):
        super(TransformerModel, self).__init__()
        self.hparams = hparams

        self.model_type = "Transformer"
        self.input_dim = 300

        self.pos_encoder = PositionalEncoding(self.input_dim, dropout = 0.2)
        encoder_layers = TransformerEncoderLayer(
            self.input_dim, hparams.num_heads, hparams.hid_dim, hparams.dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, hparams.num_layers
        )
        # self.char_embedding = nn.Embedding(hparams.vocab_size, hparams.emb_dim)
        # self.regex_embedding = nn.Embedding(hparams.regex_size, hparams.emb_dim)
        self.dropout = nn.Dropout(hparams.dropout)
        self.decoder = nn.Linear(self.input_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.regex_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, char_inputs, regex_inputs):
        char_inputs = char_inputs * math.sqrt(self.hparams.emb_dim)
        # regex_inputs = self.regex_embedding(regex_inputs.t()) * math.sqrt(self.hparams.emb_dim)

        embed = torch.cat([char_inputs], dim=2)
        # embed = char_inputs
        src = self.pos_encoder(embed.transpose(0, 1))
        output = self.transformer_encoder(src)
        output, _ = torch.max(output, dim=0)
        output = self.dropout(output)
        output = self.decoder(output)
        return torch.sigmoid(output)

    def training_step(self, batch, batch_idx):
        word_inputs, regex_inputs, labels = batch

        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, regex_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        word_inputs, regex_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, regex_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())

        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)



class TransoformerDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.generator = NCGenerator()
        self.model = TransformerModel(self.hparams.model)
        self.char_vocab = FastText()

        if hparams.gen_method == "DSL":
            logger.warning("Running DSL")
            self.generator = DSLGenerator()

    def extract_features(self, data, labels=None, retrain=False):
        regex_data = [str2regex(x, match_whole_token=False) for x in data]
        if retrain:
            # self.char_vocab = build_vocab(list(itertools.chain.from_iterable([data])))
            # self.hparams.model.vocab_size = len(self.char_vocab)
            self.regex_vocab = build_vocab(
                list(itertools.chain.from_iterable([regex_data]))
            )
            self.hparams.model.regex_size = len(self.regex_vocab)

        char_data, _ = stack_and_pad_tensors(
            [
                torch.tensor(self.char_vocab.lookup_vectors(list(str_value)))
                if str_value
                else torch.zeros(1, 300)
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

            return char_data, regex_data, label_data
        return char_data, regex_data

    def reset(self):
        self.model = TransformerModel(self.hparams.model)

    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        data, labels = self.generator.fit_transform(ec_str_pairs, values)

        feature_tensors_with_labels = self.extract_features(
            data, labels, retrain=True
        )

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, _, _ = split_train_test_dls(
            dataset,
            unzip_and_stack_tensors,
            self.hparams.model.batch_size,
            ratios=[1, 0],
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

        test_dataset = TensorDataset(*feature_tensors)

        test_dataloader =  DataLoader(
            test_dataset,
            batch_size=self.hparams.model.batch_size,
            collate_fn=unzip_and_stack_tensors,
            pin_memory=True,
            num_workers=16,
        )

        # self.model.eval()
        preds = []
        for batch in test_dataloader:
            preds.append(self.model.forward(*batch))
        return torch.cat(preds, dim=0).squeeze().detach().cpu().numpy()


class RoBertaModel(BaseModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)

        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = batch
        labels = inputs["labels"]

    
        outputs = self.forward(inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        acc = (predictions == labels).sum().float() / labels.shape[0]

        loss = outputs.loss

        logs = {"train_loss": loss, "train_acc": acc}

        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        inputs = batch
        labels = inputs["labels"]
    
        outputs = self.forward(inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        acc = (predictions == labels).sum().float() / labels.shape[0]

        loss = outputs.loss

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []        


class RoBertaDetector(TransoformerDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        self.generator = NCGenerator()

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RoBertaModel(self.hparams.model)


    def extract_features(self, data, labels=None):
        input_data = self.tokenizer.batch_encode_plus(data, padding=True, truncation=True)

        if labels is not None:
            input_data["labels"] = labels
        return input_data

    def reset(self):
        self.model = RoBertaModel(self.hparams.model)

    def idetect_values(self, ec_str_pairs: str, values: List[str]):
        data, labels = self.generator.fit_transform(ec_str_pairs, values)

        encodings = self.extract_features(
            data, labels
        )

        train_dataset = HuggingfaceDataset(encodings=encodings)

        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.model.batch_size, shuffle=True, num_workers=16)

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

        encodings = self.extract_features(values)

        test_dataset = HuggingfaceDataset(encodings=encodings)

        test_dataloader = DataLoader(test_dataset, batch_size=self.hparams.model.batch_size, shuffle=True, num_workers=16)

        # self.model.train()

        self.model.eval()
        preds = []
        for batch in test_dataloader:
            result = self.model.forward(batch).logits
            preds.append(torch.argmax(result, dim=1))
        return torch.cat(preds, dim=0).squeeze().detach().cpu().numpy()