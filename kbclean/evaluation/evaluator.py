import csv
from kbclean.recommendation.psl import PSLearner
from kbclean.datasets.dataset import Dataset
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from kbclean.evaluation.report import Report
from kbclean.recommendation.metal import MetalLeaner
from kbclean.recommendation.feedback.oracle import Oracle
from loguru import logger


class Evaluator:
    def __init__(self):
        self.name2dataset = {}

    def read_dataset(self, data_path):
        data_path = Path(data_path)

        name = data_path.name
        self.name2dataset[name] = Dataset.from_path(data_path)

    def average_report(self, *reports):
        concat_df = pd.concat(reports)
        concat_df["precision"] = concat_df["precision"] * concat_df["support"]
        concat_df["recall"] = concat_df["recall"] * concat_df["support"]
        concat_df = concat_df.groupby(level=0).sum()
        concat_df["precision"] = concat_df["precision"] / concat_df["support"]
        concat_df["recall"] = concat_df["recall"] / concat_df["support"]
        concat_df["f1-score"] = concat_df.apply(
            lambda x: 2 * x["precision"] * x["recall"] / (x["precision"] + x["recall"]),
            axis=1,
        )
        return concat_df

    def evaluate_df(self, detector, raw_df, cleaned_df, groundtruth_df):
        prob_df = detector.detect(raw_df, cleaned_df)
        return Report(prob_df, raw_df, cleaned_df, groundtruth_df)

    def ievaluate_df(
        self, detector, dataset, label_df, col2pairs
    ):
        prob_df = detector.idetect(dataset.dirty_df, label_df, col2pairs)
        return prob_df, Report(prob_df, dataset)

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

    def ievaluate(self, detector, method, dataset_path, output_path, k=1, e=5):
        output_path = Path(output_path)
        self.read_dataset(dataset_path)
        running_times = []
        name2ireport = [{} for _ in range(k)]

        for name, dataset in self.name2dataset.items():
            label_df = pd.DataFrame(
                np.ones((len(dataset.dirty_df), len(dataset.dirty_df.columns))) * -1,
                columns=dataset.dirty_df.columns,
            )
            col2pairs = defaultdict(list)

            print(f"Evaluating on {name}...")
            if all(self.name2dataset[name].groundtruth_df.stack() == True):
                continue
            logger.info(f"Evaluating on {name}...")

            oracle = Oracle(self.name2dataset[name])

            start_time = time.time()

            if detector.hparams.combine_method =="metal":
                recommender = MetalLeaner(
                    self.name2dataset[name].dirty_df, oracle, detector.hparams
                )
            else:
                recommender = PSLearner(
                    self.name2dataset[name].dirty_df, oracle, detector.hparams
                )
            print("Loading Metal time", time.time() - start_time)

            score_df = None

            for i in range(k):
                start_time = time.time()
                logger.info("----------------------------------------------------")
                logger.info(f"Running active learning step {i}...")
                for col_i, col in enumerate(self.name2dataset[name].dirty_df.columns):
                    label_df, col2pairs = recommender.next_for_each_col(
                        col, score_df, label_df, col2pairs, e
                    )

                running_time = time.time() - start_time
                logger.info(f"Total recommendation time for step {i}: {running_time}")

                start_time = time.time()

                score_df, report = self.ievaluate_df(
                    detector,
                    self.name2dataset[name],
                    label_df,
                    col2pairs,
                )

                logger.info(f"Total running time for step {i}: {time.time() - start_time}")

                running_times.append(time.time() - start_time)

                logger.info("Report:\n" + str(report.report))

                report.serialize(
                    output_path
                    / method
                    / Path(dataset_path).name
                    / Path(name).stem
                    / str(i)
                )

                name2ireport[i][name] = report.report
                score_df = report.scores
                detector.reset()

        for i in range(k):
            avg_report = self.average_report(*list(name2ireport[i].values()))
            (Path(output_path) / method / Path(dataset_path).name / "summary").mkdir(
                exist_ok=True, parents=True
            )
            avg_report.to_csv(
                Path(output_path)
                / method
                / Path(dataset_path).name
                / "summary"
                / f"{i}.csv"
            )
            logger.info(f"Step {i} average report:\n{avg_report}")
            logger.info(f"Mean running time: {np.mean(running_times)}")
        return name2ireport[k - 1]
