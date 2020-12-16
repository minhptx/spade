from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from kbclean.datasets.dataset import Dataset
from kbclean.detection.addons.highway import Highway
from kbclean.detection.base import ActiveDetector
from kbclean.detection.features.holo import HoloFeaturizer
from kbclean.transformation.noisy_channel import NegNCGenerator
from kbclean.utils.data.helpers import split_train_test_dls, unzip_and_stack_tensors
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import FastText


class HoloLearnableModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.highway = Highway(
            self.hparams.emb_dim, self.hparams.num_layers, torch.nn.functional.relu
        )
        self.linear = nn.Linear(self.hparams.emb_dim, 1)

    def forward(self, inputs):
        avg_input = torch.mean(inputs, dim=1)
        hw_out = self.highway(avg_input)
        return self.linear(hw_out)


class HoloModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.char_model = HoloLearnableModule(hparams)
        self.word_model = HoloLearnableModule(hparams)

        self.fcs = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hparams.input_dim),
            nn.Dropout(hparams.dropout),
            nn.Linear(hparams.input_dim, 1),
        )

    def forward(self, word_inputs, char_inputs, other_inputs):
        word_out = self.word_model(word_inputs)
        char_out = self.char_model(char_inputs)

        concat_inputs = torch.cat([char_out], dim=1).float()
        return torch.sigmoid(concat_inputs)

    def training_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "acc": acc, "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_idx):
        word_inputs, char_inputs, other_inputs, labels = batch
        labels = labels.view(-1, 1)
        probs = self.forward(word_inputs, char_inputs, other_inputs)
        loss = F.binary_cross_entropy(probs, labels.float())
        preds = probs >= 0.5
        acc = (labels == preds).sum().float() / labels.shape[0]
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return [optim.AdamW(self.parameters(), lr=self.hparams.lr)], []


class HoloDetector(ActiveDetector):
    def __init__(self, hparams):
        self.hparams = hparams
        self.feature_extractor = HoloFeaturizer()

        self.tokenizer = get_tokenizer("spacy")
        self.fasttext = FastText()
        self.scaler = MinMaxScaler()
        self.generator = NegNCGenerator()

    def extract_features(self, data, labels=None):
        if labels:
            features = self.scaler.fit_transform(
                self.feature_extractor.fit_transform(data)
            )
        else:
            features = self.scaler.transform(self.feature_extractor.transform(data))

        features = torch.tensor(features)

        word_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(self.tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor

        char_data = stack_and_pad_tensors(
            [
                self.fasttext.lookup_vectors(list(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor

        if labels is not None:
            label_data = torch.tensor(labels)

            return word_data, char_data, features, label_data
        return word_data, char_data, features

    def reset(self):
        self.model = HoloModel(self.hparams.model)

    def idetect_values(self, ec_str_pairs: str, values: List[str], recommender):
        data, labels = self.generator.fit_transform(
            ec_str_pairs, values, recommender.most_positives()
        )

        feature_tensors_with_labels = self.extract_features(data, labels)

        dataset = TensorDataset(*feature_tensors_with_labels)

        train_dataloader, val_dataloader, _ = split_train_test_dls(
            dataset,
            unzip_and_stack_tensors,
            self.hparams.model.batch_size,
            ratios=[0.7, 0.1],
            num_workers=1,
        )
        if len(train_dataloader) > 0:
            self.model = HoloModel(self.hparams.model)

            self.model.train()

            trainer = Trainer(
                gpus=4,
                distributed_backend="dp",
                val_percent_check=0,
                max_epochs=self.hparams.model.num_epochs,
            )
            trainer.fit(
                self.model, train_dataloader=train_dataloader,
            )

        feature_tensors = self.extract_features(values)

        self.model.eval()
        pred = self.model.forward(*feature_tensors)

        return pred.squeeze(1).detach().cpu().numpy()

    def idetect(
        self,
        dataset: Dataset,
        recommender,
        score_df: pd.DataFrame,
        label_df: pd.DataFrame,
    ):
        prediction_df = dataset.dirty_df.copy()
        for col_i, column in enumerate(dataset.dirty_df.columns):
            value_arr = label_df.iloc[:, col_i].values
            neg_indices = np.where(value_arr == -1)
            pos_indices = np.where(value_arr == 1)

            if len(neg_indices) == 0:
                logger.info(f"Skipping column {column}")
                prediction_df[column] = pd.Series(
                    [1.0 for _ in range(len(dataset.dirty_df))]
                )
            else:
                outliers = self.idetect_col(
                    dataset, recommender, column, pos_indices, neg_indices
                )
                prediction_df[column] = pd.Series(outliers)
        return prediction_df
