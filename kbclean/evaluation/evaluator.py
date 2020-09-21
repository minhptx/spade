from collections import defaultdict
import csv
from kbclean.utils.data.helpers import diff_dfs
from kbclean.recommendation.active import Uncommoner
from pathlib import Path

import numpy as np
import pandas as pd
from kbclean.evaluation.report import Report
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    def __init__(self):
        pass

    def read_dataset(self, data_path):
        data_path = Path(data_path)

        raw_path = data_path / "raw"
        cleaned_path = data_path / "cleaned"

        name2raw = {}
        name2cleaned = {}
        name2groundtruth = {}

        for file_path in list(raw_path.iterdir()):
            name = file_path.name
            name2raw[name] = pd.read_csv(
                raw_path / name, keep_default_na=False, dtype=str
            ).applymap(lambda x: x[:100])
            name2cleaned[name] = pd.read_csv(
                cleaned_path / name, keep_default_na=False, dtype=str
            ).applymap(lambda x: x[:100])

            name2groundtruth[name] = name2raw[name] == name2cleaned[name]
        return name2raw, name2cleaned, name2groundtruth

    def average_report(self, *reports):
        return pd.concat(reports).groupby(level=0).mean()

    def evaluate_df(self, detector, raw_df, cleaned_df, groundtruth_df):
        prob_df = detector.detect(raw_df, cleaned_df)
        return Report(prob_df, raw_df, cleaned_df, groundtruth_df)

    def ievaluate_df(self, detector, examples, raw_df, cleaned_df, groundtruth_df):
        prob_df = detector.idetect(raw_df, examples)
        return Report(prob_df, raw_df, cleaned_df, groundtruth_df)

    def fake_ievaluate_df(self, detector, raw_df, cleaned_df, groundtruth_df, k):
        prob_df = detector.eval_idetect(raw_df, cleaned_df, k)

        return Report(prob_df, raw_df, cleaned_df, groundtruth_df)

    def evaluate(self, detector, dataset, output_path=None):
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(dataset)

        name2report = {}

        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")

            report = self.evaluate_df(
                detector, name2raw[name], name2cleaned[name], name2groundtruth[name]
            )

            logger.info("Report:\n" + str(report.report))

            report.serialize(output_path / Path(dataset).name / Path(name).stem)

            name2report[name] = report.report

        avg_report = self.average_report(*list(name2report.values()))
        avg_report.to_csv(Path(output_path) / "summary.csv")

        logger.info("Average report:\n" + str(avg_report))

        return name2report

    def ievaluate(self, detector, dataset, output_path, step=1):
        output_path = Path(output_path)
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(dataset)

        name2ireport = defaultdict(dict)

        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")
            recommender = Uncommoner(name2raw[name], name2cleaned[name], detector.hparams)
            scores_df = None

            for i in range(step):
                logger.info("----------------------------------------------------")
                logger.info(f"Running active learning step {i}...")
                col2examples = recommender.update(i, scores_df)

                report = self.ievaluate_df(
                    detector, col2examples, name2raw[name], name2cleaned[name], name2groundtruth[name]
                )

                logger.info("Report:\n" + str(report.report))

                report.serialize(output_path / Path(dataset).name / Path(name).stem/ str(i))

                name2ireport[i][name] = report.report
                scores_df = report.scores
            detector.reset()

        for i in range(step):
            avg_report = self.average_report(*list(name2ireport[i].values()))
            (Path(output_path) / Path(dataset).name / "summary").mkdir(exist_ok=True, parents=True)
            avg_report.to_csv(Path(output_path) / Path(dataset).name / "summary" / f"{i}.csv")
            logger.info(f"Step {i} average report:\n{avg_report}")

        return name2ireport[step-1]

    def fake_ievaluate(self, detector, dataset, output_path=None):
        output_path = Path(output_path)
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(dataset)

        name2report = {}

        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")

            report = self.fake_ievaluate_df(
                detector, name2raw[name], name2cleaned[name], name2groundtruth[name], 2
            )

            logger.info("Report:\n" + str(report.report))

            report.serialize(output_path / Path(name).stem)

            name2report[name] = report.report

        avg_report = self.average_report(*list(name2report.values()))
        avg_report.to_csv(Path(output_path) / "summary.csv")

        logger.info("Average report:\n" + str(avg_report))

        return name2report
