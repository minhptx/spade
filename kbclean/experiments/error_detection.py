import click
import pandas as pd
from loguru import logger

import sys
from kbclean.detection import DistanceActiveDetector, AdhocDetector, HoloActiveDetector
from kbclean.evaluation import Evaluator
from kbclean.utils.inout import load_config

config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} - {message}"},
        {"sink": "error.log", "serialize": True, "level": "ERROR"},
        {"sink": "info.log", "serialize": True, "level": "INFO"},
    ]
}

logger.configure(**config)

@click.group()
def cli():
    pass


@cli.command()
@click.option("--data_path", "-d", help="Path to dataset")
@click.option(
    "--config_path", "-c", help="Path to configuration file", default="config"
)
@click.option(
    "--output_path", "-o", help="Path to output directory", default="output"
)
@click.option(
    "--method", "-m", help="Method for outlier detection", default="deep_clean"
)
@click.option("--interactive", "-i", help="Interactive detection", default=True)
def evaluate(data_path, config_path, output_path, method, interactive):
    evaluator = Evaluator()

    hparams = load_config(config_path)

    if method == "adhoc_clean":
        detector = AdhocDetector(hparams.adhoc)
    elif method == "distance_clean":
        detector = DistanceActiveDetector(hparams.distance)
    else:
        detector = HoloActiveDetector(hparams.holo)
    if interactive:
        evaluator.fake_ievaluate(detector, data_path, output_path)
    else:
        evaluator.evaluate(detector, data_path, output_path)


@cli.command()
@click.option("--config_path", help="Path to configuration file", default="config")
@click.option("--save_path", help="Path to save location", default="config")
@click.option("--method", help="Method for outlier detection", default="deep_clean")
def train(config_path, save_path, method):
    hparams = load_config(config_path)

    if method == "adhoc_clean":
        detector = AdhocDetector(hparams.adhoc)
    else:
        detector = ActiveDetector(hparams.bert)

    detector.prepare()


if __name__ == "__main__":
    cli()
