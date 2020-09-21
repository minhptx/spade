import os
from torch.nn.modules.rnn import LSTM
from kbclean.detection.active_transform.lstm import LSTMNaiveDetector
import random
import shutil
import sys

import click
from kbclean.detection import AdhocDetector
from kbclean.detection.active_transform import DistanceDetector, HoloDetector
from kbclean.detection.active_transform import LSTMDetector
from kbclean.evaluation import Evaluator
from kbclean.recommendation.active import Uncommoner
from kbclean.utils.inout import load_config
from loguru import logger

config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} - {message}", "level": "WARNING"},
        {
            "sink": "error.log",
            "format": "{time} - {message}",
            "level": "ERROR",
            "backtrace": True,
            "diagnose": True,
        },
        {"sink": "info.log", "format": "{time} - {message}", "level": "INFO",},
        {"sink": "debug.log", "format": "{time} - {message}", "level": "DEBUG",},
    ]
}


name2model = {
    "adhoc": AdhocDetector,
    "distance": DistanceDetector,
    "holo": HoloDetector,
    "lstm": LSTMDetector,
}

logger.configure(**config)

logger.info(
    "=========================================================================="
)

random.seed(1811)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data_path", "-d", help="Path to dataset")
@click.option(
    "--config_path", "-c", help="Path to configuration file", default="config"
)
@click.option("--output_path", "-o", help="Path to output directory", default="output")
@click.option(
    "--method", "-m", help="Method for outlier detection", default="deep_clean"
)
@click.option("--interactive", "-i", is_flag=True, help="Interactive detection")
@click.option("--num_gpus", help="Number of GPUs used", default=1)
@click.option("-k", help="Number of labeled examples", default=2)
def evaluate(data_path, config_path, output_path, method, interactive, num_gpus, k):
    evaluator = Evaluator()

    hparams = load_config(config_path)

    detector = name2model[method](getattr(hparams, method))
    getattr(hparams, method).num_gpus = 1

    if interactive:
        evaluator.ievaluate(detector, data_path, output_path, k)
    else:
        evaluator.evaluate(detector, data_path, output_path)


@cli.command()
def clear():
    shutil.rmtree("output", ignore_errors=True)
    os.remove("info.log")
    os.remove("error.log")
    os.remove("debug.log")

if __name__ == "__main__":
    cli()
