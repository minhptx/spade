from argparse import Namespace
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy
import torch
from numpy.lib.npyio import save
from pytorch_lightning.trainer.trainer import Trainer
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.models import Pooling, Transformer
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer
from transformers.configuration_bert import BertConfig

from kbclean.detection.base import BaseDetector, BaseModule
from kbclean.detection.bert import BertLanguageModel, BertModel
from kbclean.utils.data.dataset.csv import CsvDataset
from kbclean.utils.data.readers import PairedDFReader
from kbclean.utils.data.split import split_train_test_dls
from kbclean.utils.logger import MetricsTensorBoardLogger


class WordTransformer(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.hparams.vocab_size = hparams.transformer.vocab_size

        self.tokenizer = ByteLevelBPETokenizer()

        try:
            self.tokenizer = ByteLevelBPETokenizer(
                f"{hparams.save_path}/tokenizer-vocab.json",
                f"{hparams.save_path}/tokenizer-merges.txt",
            )
        except:
            self.fit_tokenizer(hparams.save_path)

        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )

        self.tokenizer.enable_truncation(max_length=self.hparams.transformer.max_length)

        self.model = BertLanguageModel(self.hparams.transformer, self.tokenizer)

        try:
            self.model = BertLanguageModel.load_from_checkpoint(
                checkpoint_path=f"{hparams.save_path}/transformer.ckpt",
                tokenizer=self.tokenizer,
            ).model.bert
        except:
            self.fit_language_model(hparams.save_path)

    def fit_tokenizer(self, save_path):
        self.tokenizer.train(
            files=[self.hparams.vocab_path],
            vocab_size=self.hparams.vocab_size,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        self.tokenizer.save_model(save_path, "tokenizer")

    def fit_language_model(self, save_path):
        csv_dataset = CsvDataset(self.hparams.vocab_path, limit=2000000)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            csv_dataset,
            self.model.encode_seq,
            self.model.hparams.batch_size,
            num_workers=32,
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
        self.trainer.save_checkpoint(f"{save_path}/transformer.ckpt")

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.model(**features)
        new_features = features.copy()
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        new_features.update(
            {
                "token_embeddings": output_tokens,
                "cls_token_embeddings": cls_tokens,
                "attention_mask": features["attention_mask"],
            }
        )

        if self.model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return new_features

    def get_word_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        self.tokenizer.enable_padding(length=self.hparams.transformer.max_length)
        self.tokenizer.enable_truncation(max_length=self.hparams.transformer.max_length)

        return self.tokenizer.encode(text).ids

    def get_sentence_features(self, tokens: List[int], pad_length: int):
        pad_seq_length = min(pad_length, self.hparams.transformer.max_length) + 5

        string = self.tokenizer.decode(tokens[:self.hparams.transformer.max_length])
        self.tokenizer.enable_padding(length=pad_seq_length)

        encoded_output = self.tokenizer.encode(string)

        sentence_features = {
            "input_ids": torch.tensor(encoded_output.ids).unsqueeze(0),
            "attention_mask": torch.tensor(encoded_output.attention_mask).unsqueeze(0),
        }
        return sentence_features

    def get_config_dict(self):
        return {}

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)


class TextTransformer(BaseModule):
    def __init__(self, hparams, pretrained=True):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.word_embedding_model = WordTransformer(hparams)

        self.pooling_model = Pooling(
            self.hparams.transformer.hidden_size,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        self.model = SentenceTransformer(
            modules=[self.word_embedding_model, self.pooling_model]
        )
        self.train_loss = losses.CosineSimilarityLoss(model=self.model)

        if not pretrained:
            self.fit(self.hparams.train_path)
            self.model.save(f"{self.hparams.save_path}/sent_transformer")

    def fit(self, train_path):
        reader = PairedDFReader(train_path)
        examples = reader.get_examples()[:5]

        sent_dataset = SentencesDataset(examples=examples, model=self.model)
        train_dataloader, val_dataloader, _ = split_train_test_dls(
            sent_dataset,
            self.model.smart_batching_collate,
            self.hparams.sent_transformer.batch_size,
        )

        self.trainer = Trainer(
            gpus=[0,1,2,3],
            distributed_backend="dp",
            logger=MetricsTensorBoardLogger("tt_logs", "deep"),
            max_epochs=1,
        )
        self.trainer.fit(
            self, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader],
        )

        self.trainer.save_checkpoint(f"{self.hparams.save_path}/model.ckpt")

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["labels"]
        x_copy = deepcopy(x)

        loss = self.train_loss(x, y.float())
        _, y_prob = self.train_loss(x_copy, None)

        y_hat = y_prob > 0.5

        acc = (y == y_hat).sum().float() / y.shape[0]

        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["labels"]
        x_copy = deepcopy(x)

        loss = self.train_loss(x, y.float())
        _, y_prob = self.train_loss(x_copy, None)

        y_hat = y_prob > 0.5
        acc = (y == y_hat).sum().float() / y.shape[0]

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.sent_transformer.lr)
        return (
            [optimizer],
            [
                self.model._get_scheduler(
                    optimizer,
                    "warmuplinear",
                    self.hparams.sent_transformer.warmup_steps,
                    self.hparams.sent_transformer.num_epochs,
                )
            ],
        )


class DeepDetector(BaseDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        try:
            self.model = TextTransformer.load_from_checkpoint(f"{hparams.save_path}/model.ckpt", hparams=self.hparams)
        except Exception as e:
            self.model = TextTransformer(self.hparams, pretrained=False)


    def prepare(self):
        pass

    def detect_values(self, values: List[str]):
        encoded_values = self.model.model.encode(values)
        mean_value = np.mean(encoded_values, axis=0)
        outliers = []
        for value in encoded_values:
            dist =  scipy.spatial.distance.cdist([value], [mean_value], "cosine")[0]
            if dist < 0.5:
                outliers.append(True)
            else:
                outliers.append(False)
        return outliers


    def detect(self, df: pd.DataFrame):
        result_df = df.copy()
        for column in df.columns:
            values = df[column].values.tolist()
            outliers = self.detect_values(values)
            result_df[column] = pd.Series(outliers)
        return result_df

    def save(self, save_path):
        self.trainer.save_checkpoint(f"{save_path}/model.ckpt")

    @staticmethod
    def load(load_path, hparams):
        detector = DeepDetector(hparams)
        detector.model = TextTransformer.load_from_checkpoint(
            checkpoint_path=f"{load_path}/model.ckpt"
        )

        return detector
