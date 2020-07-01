from argparse import ArgumentParser, Namespace

import dill as pickle
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from cleaning.detection.deep import Detector
from evaluation.evaluator import Evaluator
from models.language_modeler import BertLanguageModel, RNNLanguageModel

parser = ArgumentParser()
parser.add_argument("-i", "--input_file")
parser.add_argument("-g", "--groundtruth_file")
parser.add_argument("-e", "--encoding_model", default="seq2seq")
parser.add_argument("-r", "--error_model", default="bert")

if __name__ == "__main__":
    args = parser.parse_args()
    values = (
        pd.read_csv(args.input_file, keep_default_na=False, dtype=str)
        .iloc[:, 0]
        .apply(lambda x: "^" + x[:100])
        .values.tolist()
    )
    groundtruth = (
        pd.read_csv(args.groundtruth_file, keep_default_na=False, dtype=str)
        .iloc[:, 0]
        .apply(lambda x: "^" + x[:100])
        .values.tolist()
    )

    hparams = {"batch_size": 1000}
    hparams = Namespace(**hparams)

    # seq2seq_encoder = pickle.load(open("models/encoders/seq2seq_encoder.pkl", "rb"))

    # encoding_model = VSeq2Seq.load_from_checkpoint(
    #     checkpoint_path=f"models/{args.encoding_model}.ckpt",
    #     sos_index=seq2seq_encoder.token_to_index["^"],
    #     padding_index=seq2seq_encoder.padding_index,
    # ).cuda()

    if args.error_model == "rnn":
        lm_encoder = pickle.load(open("models/lm_encoder.pkl", "rb"))
        error_model = RNNLanguageModel.load_from_checkpoint(
            checkpoint_path=f"models/{args.error_model}.ckpt",
            padding_index=lm_encoder.padding_index,
        ).cuda()
    elif args.error_model == "bert":
        tokenizer = ByteLevelBPETokenizer("./webtables-vocab.json", "./webtables-merges.txt")

        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
        )
        hparams.vocab_size = tokenizer.get_vocab_size()
        tokenizer.enable_truncation(max_length=100)

        error_model = BertLanguageModel.load_from_checkpoint(
            checkpoint_path=f"models/{args.error_model}.ckpt", tokenizer=tokenizer
        ).cpu()

    detector = Detector(hparams, error_model)

    result = detector.detect([x for x in values])

    evaluator = Evaluator("debug.csv")
    evaluator.evaluate(values, groundtruth, result)
