import click
import pandas as pd

from kbclean.detection import BertDetector, DeepDetector, StatsDetector
from kbclean.evaluation import Evaluator
from kbclean.utils.inout import load_config


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data_path", help="Path to dataset")
@click.option("--config_path", help="Path to configuration file", default="config")
@click.option("--method", help="Method for outlier detection", default="deep_clean")
def evaluate(data_path, config_path, method):
    evaluator = Evaluator()

    hparams = load_config(config_path)

    if method == "stats_clean":
        detector = StatsDetector(hparams.stats)
    elif method == "bert_clean":
        detector = BertDetector(hparams.bert)
    else:
        detector = DeepDetector(hparams.deep)

    evaluator.evaluate(detector, data_path, output_path="output")

@cli.command()
@click.option("--config_path", help="Path to configuration file", default="config")
@click.option("--save_path", help="Path to save location", default="config")
@click.option("--method", help="Method for outlier detection", default="deep_clean")
def prepare(config_path, save_path, method):
    hparams = load_config(config_path)
    if method == "stats_clean":
        detector = StatsDetector(hparams.stats)
    elif method == "bert_clean":
        detector = BertDetector(hparams.bert)
    else:
        detector = DeepDetector(hparams.deep)

    detector.prepare()


if __name__ == "__main__":
    cli()
