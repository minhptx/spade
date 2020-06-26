import argparse
import random
from argparse import Namespace
from functools import partial

import dill as pickle
import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import Callback, Trainer
from torch.utils.data import DataLoader, random_split
from torchnlp.encoders.text import CharacterEncoder

from models.auto_encoder import VSeq2Seq
from utils.logger import MetricsTensorBoardLogger


class ReconstructionCallback(Callback):
    def __init__(self, data, collate_fn, char_encoder):
        self.collate_fn = collate_fn
        self.data = data
        self.char_encoder = char_encoder
        self.step = 0

    def on_epoch_end(self, trainer, pl_module):
        sampled_data = random.choices(self.data, k=10)
        inp, lengths, examples = self.collate_fn(sampled_data)
        dec_outputs, _ = pl_module.forward(inp.cuda(), lengths.cuda())
        best_outputs = torch.argmax(dec_outputs, dim=2)

        dec_outputs = pl_module.encode(inp.cuda(), lengths.cuda())

        pl_module.logger.experiment.add_embedding(dec_outputs,
                                                  metadata=examples,
                                                  global_step=self.step)

        reconstructed_text = ""
        for source, target in list(
                zip(self.char_encoder.batch_decode(best_outputs, lengths),
                    examples)):
            reconstructed_text += f"'{target}' --- '{source}'  \n"
        pl_module.logger.experiment.add_text("reconstruct", reconstructed_text,
                                             self.step)
        self.step += 1


def split_tran_test_dls(string_data, collate_fn, batch_size):
    train_length = int(len(string_data) * 0.7)
    val_length = int(len(string_data) * 0.2)
    train_dataset, val_dataset, test_dataset = random_split(
        list(string_data),
        [
            train_length, val_length,
            len(string_data) - train_length - val_length
        ],
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
    )

    return train_dl, val_dl, test_dl


def collate_fn_no_labels(batch, char_encoder):
    inputs, lengths = char_encoder.batch_encode(batch)
    return inputs, lengths, batch


def collate_fn_no_labels_cuda(batch, char_encoder):
    inputs, lengths = char_encoder.batch_encode(batch)
    return inputs.cuda(), lengths.cuda()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file")
parser.add_argument('-c', "--config_file", default=r"config/vseq2seq.yml")

if __name__ == "__main__":
    args = parser.parse_args()

    hparams = yaml.load(open(f'{args.config_file}', "r"),
                        Loader=yaml.FullLoader)
    hparams_seq2seq = Namespace(**hparams)
    print("Data is not encoded. Training seq2seq model...")

    data = (pd.read_csv(args.input_file,
                        keep_default_na=False).iloc[:, 0].apply(
                            lambda x: "^" + x[:100]).values.tolist())

    character_encoder = CharacterEncoder(data, append_eos=True)

    pickle.dump(character_encoder, open("models/seq2seq_encoder.pkl", "wb"))

    partial_collate_fn = partial(collate_fn_no_labels,
                                 char_encoder=character_encoder)

    train_dataloader, val_dataloader, test_dataloader = split_tran_test_dls(
        data, partial_collate_fn, hparams_seq2seq.batch_size)

    hparams_seq2seq.vocab_size = character_encoder.vocab_size

    seq2seq = VSeq2Seq(hparams_seq2seq, character_encoder.token_to_index["^"],
                       character_encoder.padding_index)

    trainer = Trainer(
        gpus=[0, 1, 2, 3],
        amp_level="O1",
        distributed_backend="dp",
        callbacks=[
            ReconstructionCallback(data, partial_collate_fn,
                                   character_encoder),
        ],
        logger=MetricsTensorBoardLogger("tt_logs", "seq2seq"),
        max_epochs=20,
    )
    trainer.fit(seq2seq,
                train_dataloader=train_dataloader,
                val_dataloaders=[val_dataloader])

    print("Finish training seq2seq model. Testing seq2seq model...")

    # print(trainer.test(encoder, test_dataloaders=[test_dataloader]))

    trainer.save_checkpoint("models/seq2seq.ckpt")

    partial_collate_fn = partial(collate_fn_no_labels_cuda,
                                 char_encoder=character_encoder)

    full_dataloader = DataLoader(
        data,
        batch_size=hparams_seq2seq.batch_size,
        collate_fn=partial_collate_fn,
    )

    print("Encoding data for GAN model...")
    encoded_batches = []

    for inputs, lengths in full_dataloader:
        encoded_batches.append(
            seq2seq.encode(inputs, lengths).detach().cpu().numpy())

    encoded_data = np.concatenate(encoded_batches, axis=0)
