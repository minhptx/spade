from kbclean.detection.active_transform.lstm_nosyntactic import LSTM2Detector
from pathlib import Path
from kbclean.detection.active_transform.random_forest import (
    RFDetector,
    SVMDetector,
    XGBDetector,
)
from kbclean.detection.active_transform.lstm_nosemantic import LSTM1Detector
from kbclean.detection.active_transform.lstm_nosemantic import LSTM1Detector
import os
import random
import shutil
import sys

import click
from kbclean.detection.active_transform import LSTMDetector
from kbclean.evaluation import Evaluator
from kbclean.utils.inout import load_config
from loguru import logger
from pandarallel import pandarallel

import neptune

pandarallel.initialize()

neptune.init(
    project_qualified_name="clapika/spade",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzYwZWY5ZjYtOTAxOS00MzhlLTlmY2EtZjRiMDkxNDhiYjQ3In0=",
)

# Create experiment

config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} - {message}", "level": "INFO"},
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
    "lstm": LSTMDetector,
    "random_forest": RFDetector,
    "xgb": XGBDetector,
    "svm": SVMDetector,
    "nosemantic": LSTM1Detector,
    "nosyntactic": LSTM2Detector,
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
@click.option("-k", help="Number of iterations", default=2)
@click.option("-e", help="Number of examples per iteration", default=2)
def evaluate(data_path, config_path, output_path, method, interactive, num_gpus, k, e):

    evaluator = Evaluator()

    debug_dir = f"debug/{Path(data_path).name}"

    hparams = load_config(config_path)

    detector = name2model[method](getattr(hparams, method))
    getattr(hparams, method).num_gpus = num_gpus
    getattr(hparams, method).num_examples = e
    getattr(hparams, method).debug_dir = debug_dir

    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    neptune.create_experiment(
        Path(data_path).name,
        params=vars(getattr(hparams, method).model),
        tags=[method],
        upload_source_files=["*.py"],
    )

    if interactive:
        evaluator.ievaluate(
            detector, method, data_path, output_path, k=k, num_examples=e
        )
    else:
        evaluator.benchmark(
            detector, method, data_path, output_path, k=k, num_examples=e
        )



@cli.command()
def clear():
    shutil.rmtree("output", ignore_errors=True)
    shutil.rmtree("debug", ignore_errors=True)
    os.remove("info.log")
    os.remove("error.log")
    os.remove("debug.log")


if __name__ == "__main__":
    cli()
