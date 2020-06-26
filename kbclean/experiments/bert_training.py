import functools
from argparse import ArgumentParser, Namespace

import torch
import yaml
from pytorch_lightning import Trainer
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence

from models.language_modeler import BertLanguageModel
from utils.data.dataset import CsvDataset
from utils.data.split import split_train_test_dls
from utils.logger import MetricsTensorBoardLogger

parser = ArgumentParser()
parser.add_argument("--input", "-i")
parser.add_argument("--config", "-c")


def collate_fn(batch, tokenizer):
    encoded_batch = tokenizer.encode_batch(batch)
    return pad_sequence([torch.tensor(x.ids) for x in encoded_batch], batch_first=True)


if __name__ == "__main__":
    args = parser.parse_args()

    hparams = yaml.load(open(f"{args.config}", "r"), Loader=yaml.FullLoader)
    hparams = Namespace(**hparams)

    # tokenizer = ByteLevelBPETokenizer()

    # tokenizer.train(
    #     files=[args.input],
    #     vocab_size=52000,
    #     min_frequency=2,
    #     special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    # )

    # tokenizer.save(".", "webtables")

    tokenizer = ByteLevelBPETokenizer("./webtables-vocab.json", "./webtables-merges.txt")

    hparams.vocab_size = tokenizer.get_vocab_size()

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=100)

    csv_dataset = CsvDataset(args.input, limit=200000)

    partial_collate_fn = functools.partial(collate_fn, tokenizer=tokenizer)

    train_dataloader, val_dataloader, test_dataloader = split_train_test_dls(
        csv_dataset, partial_collate_fn, hparams.batch_size, num_workers=16
    )

    lm = BertLanguageModel(hparams, tokenizer)

    trainer = Trainer(
        gpus=[0, 1, 2, 3],
        distributed_backend="ddp",
        logger=MetricsTensorBoardLogger("tt_logs", "lm"),
        max_epochs=3,
    )
    trainer.fit(lm, train_dataloader=train_dataloader, val_dataloaders=[val_dataloader])

    trainer.save_checkpoint("models/bert.ckpt")
