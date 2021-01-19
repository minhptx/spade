import csv
from kbclean.recommendation.psl import PSLearner
from kbclean.recommendation.psl_cluster import PSL2earner
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

    def ievaluate_df(self, detector, dataset):
        detector.idetect(dataset)
        return Report(dataset)

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

    def ievaluate(self, detector, method, dataset_path, output_path, k=1, num_examples =5):
        output_path = Path(output_path)
        self.read_dataset(dataset_path)
        running_times = []
        name2ireport = [{} for _ in range(k)]

        for name, dataset in self.name2dataset.items():
            print(f"Evaluating on {name}...")
            if all(self.name2dataset[name].groundtruth_df.stack() == True):
                continue
            logger.info(f"Evaluating on {name}...")

            start_time = time.time()

            if detector.hparams.combine_method == "metal":
                recommender = MetalLeaner(self.name2dataset[name], detector.hparams)
            elif detector.hparams.combine_method == "psl":
                recommender = PSLearner(self.name2dataset[name], detector.hparams)

            print("Loading Metal time", time.time() - start_time)

            for i in range(k):
                start_time = time.time()
                logger.info("----------------------------------------------------")
                logger.info(f"Running active learning step {i}...")
                recommender.next(num_examples)

                running_time = time.time() - start_time
                logger.info(f"Total recommendation time for step {i}: {running_time}")

                report = self.ievaluate_df(detector, self.name2dataset[name])

                logger.info(
                    f"Total running time for step {i}: {time.time() - start_time}"
                )

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


    def benchmark(self, detector, method, dataset_path, output_path, k=1, num_examples =5):
        output_path = Path(output_path)
        self.read_dataset(dataset_path)
        running_times = []
        name2ireport = [{} for _ in range(k)]

        for name, dataset in self.name2dataset.items():
            print(f"Evaluating on {name}...")
            if all(self.name2dataset[name].groundtruth_df.stack() == True):
                continue
            logger.info(f"Evaluating on {name}...")

            start_time = time.time()

            if detector.hparams.combine_method == "metal":
                recommender = MetalLeaner(self.name2dataset[name], detector.hparams)
            elif detector.hparams.combine_method == "psl":
                recommender = PSLearner(self.name2dataset[name], detector.hparams)

            print("Loading Metal time", time.time() - start_time)

            for i in range(k):
                start_time = time.time()
                logger.info("----------------------------------------------------")
                logger.info(f"Running active learning step {i}...")
                recommender.next(num_examples)

                running_time = time.time() - start_time
                logger.info(f"Total recommendation time for step {i}: {running_time}")

            report = self.ievaluate_df(detector, self.name2dataset[name])

            logger.info(
                f"Total running time for step {i}: {time.time() - start_time}"
            )

            running_times.append(time.time() - start_time)

            logger.info("Report:\n" + str(report.report))

            report.serialize(
                output_path
                / method
                / Path(dataset_path).name
                / Path(name).stem
                / str(0)
            )

            name2ireport[0][name] = report.report

        for i in range(k):
            if i == 0:
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
