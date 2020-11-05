import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd
from kbclean.evaluation.report import Report
from kbclean.recommendation.active import ActiveLearner
from kbclean.utils.data.helpers import diff_dfs
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    def __init__(self):
        pass

    def read_dataset(self, data_path, data_range):
        data_path = Path(data_path)

        raw_path = data_path / "raw"
        cleaned_path = data_path / "cleaned"

        name2raw = {}
        name2cleaned = {}
        name2groundtruth = {}

        if data_range[0] == None:
            data_range[0] = 0
        if data_range[1] == None:
            data_range[1] = len(list(raw_path.iterdir()))

        for file_path in sorted(list(raw_path.iterdir()))[
            data_range[0] : data_range[1]
        ]:
            name = file_path.name
            name2raw[name] = pd.read_csv(
                raw_path / name, keep_default_na=False, dtype=str
            ).applymap(lambda x: ftfy.fix_text(x[:100]))
            name2cleaned[name] = pd.read_csv(
                cleaned_path / name, keep_default_na=False, dtype=str
            ).applymap(lambda x: ftfy.fix_text(x[:100]))
            name2groundtruth[name] = name2raw[name] == name2cleaned[name]
        return name2raw, name2cleaned, name2groundtruth

    def average_report(self, *reports):
        concat_df = pd.concat(reports)
        concat_df["precision"] = concat_df["precision"]  * concat_df["support"]
        concat_df["recall"] = concat_df["recall"] * concat_df["support"]
        concat_df = concat_df.groupby(level=0).sum()
        concat_df["precision"] = concat_df["precision"] / concat_df["support"]
        concat_df["recall"] = concat_df["recall"] / concat_df["support"]
        concat_df["f1-score"] = concat_df.apply(lambda x: 2 * x["precision"] * x["recall"] / (x["precision"] + x["recall"]), axis=1)
        return concat_df

    def evaluate_df(self, detector, raw_df, cleaned_df, groundtruth_df):
        prob_df = detector.detect(raw_df, cleaned_df)
        return Report(prob_df, raw_df, cleaned_df, groundtruth_df)

    def ievaluate_df(self, detector, recommender, raw_df, cleaned_df, groundtruth_df, score_df):
        prob_df = detector.idetect(raw_df, score_df, recommender)
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

    def ievaluate(
        self, detector, method, dataset, output_path, step=1, data_range=[0, None]
    ):
        output_path = Path(output_path)
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(
            dataset, data_range
        )
        running_times = []
        name2ireport = defaultdict(dict)

        for name in name2raw.keys():
            print(f"Evaluating on {name}...")
            if all(name2groundtruth[name].stack() == True):
                continue
            logger.info(f"Evaluating on {name}...")
            recommender = ActiveLearner(
                name2raw[name], name2cleaned[name], detector.hparams
            )

            score_df = None

            for i in range(step):
                start_time = time.time()
                logger.info("----------------------------------------------------")
                logger.info(f"Running active learning step {i}...")
                for col in name2raw[name].columns:
                    recommender.update(i, col, score_df)
                running_time = time.time() - start_time
                logger.info(
                    f"Total recommendation time for step {i}: {running_time}"
                )

                start_time = time.time()


                start_time = time.time()

                report = self.ievaluate_df(
                    detector,
                    recommender,
                    name2raw[name],
                    name2cleaned[name],
                    name2groundtruth[name],
                    score_df
                )

                logger.info(
                    f"Total running time for step {i}: {running_time}"
                )

                running_times.append(running_time)

                logger.info("Report:\n" + str(report.report))

                report.serialize(
                    output_path / method / Path(dataset).name / Path(name).stem / str(i)
                )


                name2ireport[i][name] = report.report
                score_df = report.scores
                detector.reset()

        for i in range(step):
            avg_report = self.average_report(*list(name2ireport[i].values()))
            (Path(output_path) / method / Path(dataset).name / "summary").mkdir(
                exist_ok=True, parents=True
            )
            avg_report.to_csv(
                Path(output_path) / method / Path(dataset).name / "summary" / f"{i}.csv"
            )
            logger.info(f"Step {i} average report:\n{avg_report}")
            logger.info(f"Mean running time: {np.mean(running_times)}")
        return name2ireport[step - 1]

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
