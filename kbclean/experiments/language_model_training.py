import re
from argparse import ArgumentParser, Namespace
from functools import partial

import numpy as np
import pandas as pd
import yaml
from pytorch_lightning import Trainer

from models.language_modeler import RNNLanguageModel
from utils.data.dataset import CsvDataset
from utils.data.split import split_train_test_dls
from utils.logger import MetricsTensorBoardLogger


def regexize(str_):
    str_ = "^" + str_[:100]
    str_ = re.sub(r"[0-9]", "0", str_)
    str_ = re.sub(r"[A-Z]", "A", str_)
    str_ = re.sub(r"[a-z]", "a", str_)
    return str_


def collate_fn_no_labels(batch, char_encoder):
    inputs, lengths = char_encoder.batch_encode(batch)
    return inputs, lengths, batch


parser = ArgumentParser()
parser.add_argument("-i", "--input_file", default="data/train/webtables/all.csv")
parser.add_argument("-c", "--config_file", default="config/language_model.yml")
parser.add_argument("-t", "--test_file")

if __name__ == "__main__":
    args = parser.parse_args()

    hparams = yaml.load(open(f"{args.config_file}", "r"), Loader=yaml.FullLoader)
    hparams_seq2seq = Namespace(**hparams)
    print("Reading data...")

    csv_dataset = CsvDataset(args.input_file, regexize)

    # pickle.dump(regex_encoder, open("models/encoders/regex_encoder.pkl", "wb"))

    partial_collate_fn = partial(collate_fn_no_labels, char_encoder=csv_dataset.encoder)

    train_dataloader, val_dataloader, test_dataloader = split_train_test_dls(
        csv_dataset, partial_collate_fn, hparams_seq2seq.batch_size, num_workers=16
    )

    hparams_seq2seq.vocab_size = csv_dataset.encoder.vocab_size

    print("Training language model...")

    lm = RNNLanguageModel(hparams_seq2seq, csv_dataset.encoder)

    trainer = Trainer(
        gpus=[0, 1, 2, 3],
        distributed_backend="ddp",
        num_sanity_val_steps=0,
        logger=MetricsTensorBoardLogger("tt_logs", "lm"),
        max_epochs=50,
    )
    trainer.fit(lm, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader])

    trainer.save_checkpoint("models/language_model.ckpt")
