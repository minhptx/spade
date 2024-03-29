from kbclean.detection.base import ActiveDetector
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from kbclean.detection.features.base import UnionFeaturizer
from kbclean.detection.features.embedding import CharAvgFT, WordAvgFT
from kbclean.detection.features.statistics import StatsFeaturizer, TfidfFeaturizer
from kbclean.transformation.error import ErrorGenerator
from kbclean.utils.data.helpers import split_train_test_dls, unzip_and_stack_tensors
from loguru import logger
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset


class Reducer(nn.Module):
    def __init__(self, input_dim, reduce_dim):
        super(Reducer, self).__init__()

        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.model = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2), nn.ReLU(), nn.Linear(input_dim, reduce_dim),
        )

    def forward(self, inputs):
        if inputs.shape[0] != 1:
            inputs = self.batch_norm(inputs)
        return self.model(inputs)


class LSTMModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.reducers = nn.ModuleList()

        for feature_dim in self.hparams.feature_dims:
            self.reducers.append(Reducer(feature_dim, self.hparams.reduce_dim))

        self.batch_norm = nn.BatchNorm1d(
            self.hparams.reduce_dim * len(self.hparams.feature_dims)
        )

        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.hparams.reduce_dim * len(self.hparams.feature_dims), 1,),
        )

    def forward(self, words, chars, statistics):
        concat_inputs = []

        for idx, inputs in enumerate([words, chars, statistics]):
            concat_inputs.append(self.reducers[idx](inputs.float()))

        attn_outputs = torch.cat(concat_inputs, dim=1)

        if attn_outputs.shape[0] != 1:
            attn_outputs = self.batch_norm(attn_outputs)

        return torch.sigmoid(self.model(attn_outputs.float()))

    def training_step(self, batch, batch_idx):
        (words, chars, features, labels) = batch
        labels = labels.view(-1, 1)
        weights = torch.zeros_like(labels).type_as(labels).float()
        weights[labels <= 0.5] = (labels >= 0.5).sum().float() / labels.shape[0]
        weights[labels >= 0.5] = (labels <= 0.5).sum().float() / labels.shape[0]

        probs = self.forward(words, chars, features)

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels.long() == preds).sum().float() / labels.shape[0]

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (words, chars, features, labels) = batch
        labels = labels.view(-1, 1)
        weights = torch.zeros_like(labels).type_as(labels).float()
        weights[labels <= 0.5] = (labels >= 0.5).sum().float() / labels.shape[0]
        weights[labels >= 0.5] = (labels <= 0.5).sum().float() / labels.shape[0]

        probs = self.forward(words, chars, features)

        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels.long() == preds).sum().float() / labels.shape[0]

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class LSTMDetector(ActiveDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = UnionFeaturizer(
            {"word": WordAvgFT(), "char": CharAvgFT(), "stats": StatsFeaturizer()}
        )
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

        # output_pd = train_dirty_df[[col]].copy()
        # output_pd["label"] = train_label_df[col].values.tolist()
        # output_pd["rule"] = rules
        # output_pd.to_csv(f"{self.hparams.debug_dir}/{col}_debug.csv")

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

        num_epochs = 5 if (50000 // len(train_data)) < 5 else (50000 // len(train_data))

        print(f"Training for {num_epochs} epochs")

        self.model.train()

        if len(train_dataloader) > 0:
            os.environ["MASTER_PORT"] = str(random.randint(49152, 65535))

            trainer = Trainer(
                gpus=self.hparams.gpus,
                accelerator="dp",
                max_epochs=num_epochs,
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
            start_time = time.time()
            value_arr = dataset.label_df.iloc[:, col_i].values
            neg_indices = np.where(np.logical_and(0 <= value_arr, value_arr <= 0.5))[
                0
            ].tolist()
            pos_indices = np.where(value_arr >= 0.5)[0].tolist()
            examples = [dataset.dirty_df[col][i] for i in neg_indices + pos_indices]

            logger.info(
                f"Column {col} has {len(neg_indices)} negatives and {len(pos_indices)} positives"
            )

            print("Preparation time: ", time.time() - start_time)
            start_time = time.time()

            if len(neg_indices) == 0:
                dataset.prediction_df[col] = pd.Series(
                    [1.0 for _ in range(len(dataset.dirty_df))]
                )
            else:
                logger.info(f"Detecting column {col} with {len(examples)} examples")

                pd.DataFrame(dataset.col2labeled_pairs[col]).to_csv(
                    f"{self.hparams.debug_dir}/{col}_chosen.csv"
                )

                outliers = self.idetect_col(dataset, col, pos_indices, neg_indices)
                outliers[
                    np.where(dataset.soft_label_df.loc[:, col].values == 1)[0]
                ] = 1
                outliers[
                    np.where(dataset.soft_label_df.loc[:, col].values == 0)[0]
                ] = 0

                df = pd.DataFrame(dataset.dirty_df[col].values)
                df["result"] = outliers
                df["training_label"] = dataset.label_df[col].values.tolist()
                df.to_csv(f"{self.hparams.debug_dir}/{col}_prediction.csv", index=None)
                dataset.prediction_df[col] = pd.Series(outliers)
                print("Detection time: ", time.time() - start_time)

