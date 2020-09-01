import random

import click
from loguru import logger
import sys
from kbclean.detection import (AdhocDetector, DistanceDetector,
                               HoloDetector, LSTMDetector)
from kbclean.detection.active.holo import HoloActiveLearner
from kbclean.evaluation import Evaluator
from kbclean.utils.inout import load_config

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
    "lstm": LSTMDetector
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
def evaluate(data_path, config_path, output_path, method, interactive):
    evaluator = Evaluator()

    hparams = load_config(config_path)

    detector = name2model[method](getattr(hparams, method))
    getattr(hparams, method).num_gpus = 1 

    if interactive:
        evaluator.ievaluate(detector, data_path, output_path)
    else:
        evaluator.evaluate(detector, data_path, output_path)


@cli.command()
@click.option("--data_path", "-d", help="Path to dataset")
@click.option(
    "--config_path", "-c", help="Path to configuration file", default="config"
)
def evaluate_active(data_path, config_path):
    evaluator = Evaluator()
    hparams = load_config(config_path)

    active_learner = HoloActiveLearner(hparams.holo)

    print(evaluator.evaluate_active(active_learner, data_path, 3))


if __name__ == "__main__":
    cli()
