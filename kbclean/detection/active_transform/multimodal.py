import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from kbclean.detection.active_transform.holo import HoloDetector
from kbclean.detection.features.base import UnionFeaturizer
from kbclean.detection.features.embedding import CharSeqFT
from kbclean.transformation.error import ErrorGenerator
from kbclean.utils.data.helpers import split_train_test_dls, unzip_and_stack_tensors
from loguru import logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import TensorDataset


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)

        self.W = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(1))
        self.feature_dim = feature_dim

    def forward(self, x):
        eij = (
            torch.mm(x.view(-1, self.feature_dim), self.W).view(-1, x.shape[1]) + self.b
        )

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, dim=1)


class LSTMModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.dropout = nn.Dropout(0.3)

        self.char_lstm = nn.GRU(
            300, self.hparams.char_hid_dim, batch_first=True, bidirectional=True
        )

        # self.word_lstm = nn.GRU(
        #     300, self.hparams.word_hid_dim, batch_first=True, bidirectional=True
        # )

        self.char_attention = Attention(self.hparams.char_hid_dim * 2)

        # self.word_attention = Attention(self.hparams.word_hid_dim * 2)

        self.fc = nn.Linear(
            2 * (self.hparams.char_hid_dim), 1,
        )

    def forward(self, char_inputs, char_lengths):
        packed_inputs = pack_padded_sequence(
            char_inputs, char_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, h = self.char_lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        h = torch.cat([h[0:1, :, :], h[1:, :, :]], dim=2)

        char_attn_outputs = self.char_attention(outputs)

        attn_outputs = torch.cat([char_attn_outputs], dim=1)

        return torch.sigmoid(self.fc(attn_outputs))

    def training_step(self, batch, batch_idx):
        char_inputs, char_lengths, labels = batch
        labels = labels.view(-1, 1)
        weights = torch.zeros_like(labels).type_as(labels).float()
        weights[labels <= 0.5] = (labels >= 0.5).sum().float() / labels.shape[0]
        weights[labels >= 0.5] = (labels <= 0.5).sum().float() / labels.shape[0]

        probs = self.forward(char_inputs, char_lengths)

        loss = F.binary_cross_entropy(probs, labels.float(), weights)
        preds = probs >= 0.5
        acc = (labels.long() == preds).sum().float() / labels.shape[0]

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        char_inputs, char_lengths, labels = batch
        labels = labels.view(-1, 1)
        weights = torch.zeros_like(labels).type_as(labels).float()
        weights[labels <= 0.5] = (labels >= 0.5).sum().float() / labels.shape[0]
        weights[labels >= 0.5] = (labels <= 0.5).sum().float() / labels.shape[0]

        probs = self.forward(char_inputs, char_lengths)

        loss = F.binary_cross_entropy(probs, labels.float(), weights)
        preds = probs >= 0.5
        acc = (labels.long() == preds).sum().float() / labels.shape[0]

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class LSTMDetector(HoloDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = UnionFeaturizer({"char_ft": CharSeqFT()})
        self.training = True

    def extract_features(self, dirty_df, label_df, col):
        if self.training:
            features = self.feature_extractor.fit_transform(dirty_df, col)
        else:
            features = self.feature_extractor.transform(dirty_df, col)

        self.hparams.model.feature_dims = [
            extractor.n_features(dirty_df)
            for extractor in self.feature_extractor.name2extractor.values()
        ]

        if self.training:
            labels = label_df[col]
            return features + [torch.tensor(labels.values.tolist())]
        return features

    def reset(self):
        try:
            self.model = LSTMModel(self.hparams.model)
        except:
            pass

    def idetect_col(self, dataset, col, pos_indices, neg_indices):
        generator = ErrorGenerator()
        start_time = time.time()
        self.training = True

        logger.info("Transforming data ....")
        train_dirty_df, train_label_df, rules = generator.fit_transform(
            dataset, col, pos_indices, neg_indices
        )

        output_pd = train_dirty_df[[col]].copy()
        output_pd["label"] = train_label_df[col].values.tolist()
        output_pd["rule"] = rules
        output_pd.to_csv(f"{self.hparams.debug_dir}/{col}_debug.csv")

        logger.info(f"Total transformation time: {time.time() - start_time}")

        feature_tensors_with_labels = self.extract_features(
            train_dirty_df, train_label_df, col
        )

        start_time = time.time()

        self.model = LSTMModel(self.hparams.model)

        train_data = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, _, _ = split_train_test_dls(
            train_data,
            unzip_and_stack_tensors,
            self.hparams.model.batch_size,
            ratios=[1.0, 0.0],
            num_workers=0,
            pin_memory=False,
        )

        self.model.train()

        if len(train_dataloader) > 0:
            os.environ["MASTER_PORT"] = str(random.randint(49152, 65535))

            trainer = Trainer(
                gpus=[2],
                accelerator="ddp",
                max_epochs=self.hparams.model.num_epochs,
                checkpoint_callback=False,
                logger=False,
            )

            trainer.fit(
                self.model, train_dataloader=train_dataloader,
            )
        logger.info(f"Total training time: {time.time() - start_time}")

        start_time = time.time()

        self.model.eval()
        self.training = False
        feature_tensors = self.extract_features(dataset.dirty_df, None, col)

        pred = self.model.forward(*feature_tensors)

        result = pred.squeeze().detach().cpu().numpy()
        logger.info(f"Total prediction time: {time.time() - start_time}")

        return result

    def idetect(self, dataset: pd.DataFrame):
        for col_i, col in enumerate(dataset.dirty_df.columns):
            value_arr = dataset.label_df.iloc[:, col_i].values
            neg_indices = np.where(np.logical_and(0 <= value_arr, value_arr <= 0.5))[
                0
            ].tolist()
            pos_indices = np.where(value_arr >= 0.5)[0].tolist()
            examples = [dataset.dirty_df[col][i] for i in neg_indices + pos_indices]

            logger.info(
                f"Column {col} has {len(neg_indices)} negatives and {len(pos_indices)} positives"
            )

            if len(neg_indices) == 0:
                dataset.prediction_df[col] = [1.0 for _ in range(len(dataset.dirty_df))]

            else:
                logger.info(f"Detecting column {col} with {len(examples)} examples")

                pd.DataFrame(dataset.col2labeled_pairs[col]).to_csv(
                    f"{self.hparams.debug_dir}/{col}_chosen.csv"
                )

                outliers = self.idetect_col(dataset, col, pos_indices, neg_indices)

                df = pd.DataFrame(dataset.dirty_df[col].values)
                df["result"] = outliers
                df["training_label"] = dataset.label_df[col].values.tolist()
                df.to_csv(f"{self.hparams.debug_dir}/{col}_prediction.csv", index=None)
                dataset.prediction_df[col] = outliers
