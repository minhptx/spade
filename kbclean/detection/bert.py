import pickle
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.util import pad_sequence
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer import Trainer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, BertForMaskedLM, BertPreTrainedModel

from kbclean.detection.base import BaseDetector, BaseModule
from kbclean.utils.data.dataset import CsvDataset
from kbclean.utils.data.dataset.csv import CsvDataset
from kbclean.utils.data.split import split_train_test_dls
from kbclean.utils.logger import MetricsTensorBoardLogger


def clean_str(x):
    x = x.strip().encode("ascii", "ignore").decode("ascii")
    return str2regex(x)


def str2regex(x):
    if x is None:
        return ""
    x = re.sub(r"[A-Z]", "A", x)
    x = re.sub(r"[a-z]", "a", x)
    x = re.sub(r"[0-9]", "0", x)
    return x


class BertModel(BaseModule):
    def __init__(self, hparams, tokenizer) -> None:
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer

        self.model: BertPreTrainedModel

    def encode_seq(self, batch):
        encoded_batch = self.tokenizer.encode_batch(batch)
        result = pad_sequence(
            [torch.tensor(x.ids) for x in encoded_batch], batch_first=True
        )
        return result

    def forward(self, inputs):
        return self.model.forward(inputs.long(), labels=inputs.long())

    def encode(self, inputs):
        encoded_batch = self.tokenizer.encode_batch(inputs)
        encoded_inputs = pad_sequence(
            [torch.tensor(x.ids) for x in encoded_batch], batch_first=True
        )
        return torch.mean(self.model.bert.forward(encoded_inputs)[0], dim=1)

    def training_step(self, batch, batch_idx):
        loss, scores = self.forward(batch)[:2]

        preds = torch.flatten(torch.argmax(scores, dim=2))
        true_labels = torch.flatten(batch)

        acc = (preds == true_labels).float().mean()
        loss = loss.float() / batch.shape[0]

        logs = {"train_loss": loss, "train_acc": acc}

        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        loss, scores = self.forward(batch)[:2]

        preds = torch.flatten(torch.argmax(scores, dim=2))
        true_labels = torch.flatten(batch)

        acc = (preds == true_labels).float().mean()

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.Adam(self.parameters(), lr=self.hparams.lr)]


class BertLanguageModel(BertModel):
    def __init__(self, hparams, tokenizer):
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, tokenizer)

        self.model = BertForMaskedLM(
            BertConfig(
                vocab_size=hparams.vocab_size,
                hidden_size=hparams.hidden_size,
                num_hidden_layers=hparams.num_hidden_layers,
                intermediate_size=hparams.intermediate_size,
                hidden_dropout_prob=hparams.hidden_dropout_prob,
                num_attention_heads=hparams.num_attention_heads,
            )
        )


class BertPipeline:
    def __init__(self, hparams):
        self.hparams = hparams
        self.tokenizer = ByteLevelBPETokenizer()
        self.model = BertLanguageModel(self.hparams.language_model, self.tokenizer)

    def fit(self, train_path):
        pass

    def save(self, save_path):
        self.trainer.save_checkpoint(f"{save_path}/model.ckpt")
        self.tokenizer.save_model(save_path, "tokenizer")

    @staticmethod
    def load(load_path, hparams):
        pipeline = BertPipeline(hparams)
        pipeline.tokenizer = ByteLevelBPETokenizer(
            f"{load_path}/tokenizer-vocab.json", f"{load_path}/tokenizer-merges.txt",
        )
        pipeline.model = BertLanguageModel.load_from_checkpoint(
            checkpoint_path=f"{load_path}/model.ckpt", tokenizer=pipeline.tokenizer,
        )

        return pipeline


class BertDetector(BaseDetector):
    def __init__(self, hparams):
        self.hparams = hparams

        if self.hparams.detection_method == "RF":
            self.outlier_model = IsolationForest(n_jobs=32)
        elif self.hparams.detection_method == "SVM":
            self.outlier_model = OneClassSVM()

        self.model = BertLanguageModel(hparams)

    def prepare(self, save_path):
        self.tokenizer.train(
            files=[self.hparams.vocab_path],
            vocab_size=self.hparams.vocab_size,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        self.hparams.vocab_size = self.tokenizer.get_vocab_size()

        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )

        self.tokenizer.enable_truncation(max_length=self.hparams.max_length)
        self.tokenizer.save_model(save_path)

    def fit(self, train_path):
        csv_dataset = CsvDataset(train_path, limit=2000000)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            csv_dataset, None, self.hparams.batch_size, num_workers=32
        )

        self.trainer = Trainer(
            gpus=[0, 1, 2, 3],
            distributed_backend="ddp",
            logger=MetricsTensorBoardLogger("tt_logs", "bert"),
            max_epochs=2,
        )
        self.trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=[val_dataloader],
        )

    def detect_values(self, strings: List[str]):
        tensors = []

        patterns = list(map(clean_str, strings))

        for i in range(0, len(strings), self.hparams.batch_size):
            tensor = self.bert_pipeline.model.encode(
                patterns[i : i + self.hparams.batch_size]
            )
            tensors.append(tensor)

        probs = torch.cat(tensors, dim=0).detach().cpu().numpy()
        preds = self.outlier_model.fit_predict(probs)
        return preds != -1

    def detect(self, df):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            outliers = self.detect_values(values)
            result_df[column] = pd.Series(outliers)
        return result_df

    def save(self, save_path):
        self.bert_pipeline.save(save_path)
        pickle.dump(self.outlier_model, open(f"{save_path}/detector.pkl"))

    @staticmethod
    def load(load_path, hparams):
        detector = BertDetector(hparams)
        detector.bert_pipeline = BertPipeline.load(load_path)
        detector.outlier_model = pickle.load(f"{save_path}/detector.pkl")
        return detector
