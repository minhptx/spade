import os
from pathlib import Path

import click
from numpy.core.multiarray import result_type
import pandas as pd
import rapidjson as json

from kbclean.detection.active.lstm import LSTMDetector
from kbclean.utils.inout import load_config
from loguru import logger
import sys

__PATH__ = Path(os.path.abspath(__file__))

config = {
    "handlers": [
        {"sink": sys.stdout, "level": "INFO"}
    ]
}


logger.configure(**config)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input_path", "-i", help="Path to input file", default="demo/beers.csv")
@click.option(
    "--example_path", "-e", help="Path to examples", default="demo/examples.json"
)
@click.option(
    "--output_path", "-o", help="Path to output file", default="demo/output.csv"
)
@click.option(
    "--num_gpus", help="Number of GPUs used for training", default=1
)
def detect(input_path, example_path, output_path, num_gpus):
    print("Initializing model...")
    hparams = load_config(__PATH__.parent.parent / "config" / "lstm")
    hparams.num_gpus = num_gpus
    detector = LSTMDetector(hparams=hparams)
    col2examples = json.load(open(example_path, "r"))

    df = pd.read_csv(input_path, keep_default_na=False, dtype=str)
    print("Detecting outliers...")

    result_df = detector.idetect(df, col2examples)
    for col in result_df.columns:
        result_df[col] = list(zip(df[col], result_df[col]))

    result_df = result_df.applymap(lambda x: x[0] if x[1] >= 0.5 else f"<<<{x[0]}>>>")
    result_df.to_csv(output_path, index=False)
    print(f"Finish outlier detection. Annotated output can be found in {output_path}")

if __name__ == "__main__":
    cli()