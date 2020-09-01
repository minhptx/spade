import csv
from pathlib import Path

import numpy as np
import pandas as pd
from kbclean.utils.data.helpers import diff_dfs, not_equal
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
            name2raw[name] = (
                pd.read_csv(raw_path / name, keep_default_na=False, dtype=str)
                .applymap(lambda x: x[:100])
            )
            name2cleaned[name] = (
                pd.read_csv(cleaned_path / name, keep_default_na=False, dtype=str)
                .applymap(lambda x: x[:100])
            )

            name2groundtruth[name] = name2raw[name] == name2cleaned[name]
        return name2raw, name2cleaned, name2groundtruth

    def average_report(self, *reports):
        return pd.concat(reports).groupby(level=0).mean()

    def debug(self, raw_df, cleaned_df, groundtruth_df, result_df, prob_df):
        def get_prob(x):
            return prob_df.loc[x["id"], x["col"]]

        fn_df = diff_dfs(raw_df, cleaned_df)
        fn_df["prediction"] = fn_df.apply(get_prob, axis=1)
        fn_df = fn_df[fn_df["prediction"] >= 0.5]

        diff_mask = not_equal(groundtruth_df, result_df)
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]

        changed.index.names = ["id", "col"]
        difference_locations = np.where(diff_mask)
        changed_from = raw_df.values[difference_locations]
        changed_to = cleaned_df.values[difference_locations]
        fp_df = pd.DataFrame(
            {"from": changed_from, "to": changed_to}, index=changed.index
        )
        fp_df["prediction"] = 0.0
        fp_df["id"] = fp_df.index.get_level_values("id")
        fp_df["col"] = fp_df.index.get_level_values("col")
        fp_df["prediction"] = fp_df.apply(get_prob, axis=1)
        fp_df = fp_df[fp_df["prediction"] <= 0.5]

        concat_df = pd.concat([fp_df, fn_df], ignore_index=True)

        score_df = pd.DataFrame()
        score_sr = raw_df.stack()
        score_df["data"] = score_sr.values.tolist()
        score_df.index = score_sr.index
        score_df.index.names = ["id", "col"]

        score_df["id"] = score_df.index.get_level_values("id")
        score_df["col"] = score_df.index.get_level_values("col")
        score_df["score"] = prob_df.stack().values.tolist()
        return concat_df, score_df

    def log_outputs(self, report, matrix, debug, scores, path):
        report_path = Path(path) / "report.csv"
        debug_path = Path(path) / "debug.csv"
        matrix_path = Path(path) / "matrix.csv"
        score_path = Path(path) / "scores.csv"

        path.mkdir(parents=True, exist_ok=True)

        report["index"] = report.index
        report.to_csv(report_path, index=False, quoting=csv.QUOTE_ALL)

        debug.to_csv(debug_path, index=False, quoting=csv.QUOTE_ALL)

        matrix["index"] = pd.Series(["True", "False"])
        matrix.to_csv(matrix_path, index=None, quoting=csv.QUOTE_ALL)

        scores.to_csv(score_path, index=None, quoting=csv.QUOTE_ALL)

    def evaluate_df(self, detector, raw_df, cleaned_df, groundtruth_df):
        prob_df = detector.fake_idetect(raw_df, cleaned_df)
        detected_df = prob_df.applymap(lambda x: x >= 0.5)

        flat_result = detected_df.stack().values.tolist()
        ground_truth = groundtruth_df.stack().values.tolist()

        report = pd.DataFrame(
            classification_report(ground_truth, flat_result, output_dict=True)
        ).transpose()

        matrix = pd.DataFrame(
            confusion_matrix(ground_truth, flat_result, labels=[True, False]),
            columns=["True", "False"],
        )

        debug = self.debug(raw_df, cleaned_df, groundtruth_df, detected_df, prob_df)

        return report, matrix, debug

    def step_ievaluate_df(self, detector, raw_df, cleaned_df, groundtruth_df):
        prob_df = detector.eval_idetect(raw_df, cleaned_df)
        detected_df = prob_df.applymap(lambda x: x >= 0.5)

        flat_result = detected_df.stack().values.tolist()
        ground_truth = groundtruth_df.stack().values.tolist()

        report = pd.DataFrame(
            classification_report(ground_truth, flat_result, output_dict=True)
        ).transpose()

        matrix = pd.DataFrame(
            confusion_matrix(ground_truth, flat_result, labels=[True, False]),
            columns=["True", "False"],
        )

        debug, scores = self.debug(
            raw_df, cleaned_df, groundtruth_df, detected_df, prob_df
        )

        return report, matrix, debug, scores

    def evaluate_active(self, active_learner, dataset, k):
        name2raw, _, name2groundtruth = self.read_dataset(dataset)
        true_count = 0
        for name in name2groundtruth.keys():
            active_learner.fit(name2raw[name])
            indices = active_learner.next(k)
            for col_index, row_index in indices:
                if name2groundtruth[name].iloc[row_index, col_index] == False:
                    true_count += 1

        return true_count * 1.0 / (k * len(name2groundtruth))

    def evaluate(self, detector, dataset, output_path=None):
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(dataset)

        name2report = {}

        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")

            report, matrix, debug, scores = self.evaluate_df(
                detector, name2raw[name], name2cleaned[name], name2groundtruth[name]
            )

            logger.info("Report:\n" + str(report))

            self.log_outputs(
                report, matrix, debug, scores, output_path / Path(name).stem
            )

            name2report[name] = report

        avg_report = self.average_report(*list(name2report.values()))
        avg_report.to_csv(Path(output_path) / "summary.csv")

        logger.info("Average report:\n" + str(avg_report))

        return name2report

    def ievaluate(self, detector, dataset, output_path, k=1):
        output_path = Path(output_path)
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(dataset)

        name2report = {}

        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")

            for i in range(k):
                report, matrix, debug, scores = self.step_ievaluate_df(
                    detector, name2raw[name], name2cleaned[name], name2groundtruth[name]
                )

                logger.info("Report:\n" + str(report))

                self.log_outputs(
                    report,
                    matrix,
                    debug,
                    scores,
                    output_path / str(i) / Path(name).stem,
                )
                name2report[name] = report

        avg_report = self.average_report(*list(name2report.values()))
        avg_report.to_csv(Path(output_path) / "summary.csv")
        logger.info("Average report:\n" + str(avg_report))

        return name2report

    def fake_ievaluate(self, detector, dataset, output_path=None):
        output_path = Path(output_path)
        name2raw, name2cleaned, name2groundtruth = self.read_dataset(dataset)

        name2report = {}

        for name in name2raw.keys():
            logger.info(f"Evaluating on {name}...")

            report, matrix, debug, scores = self.step_ievaluate_df(
                detector, name2raw[name], name2cleaned[name], name2groundtruth[name]
            )

            logger.info("Report:\n" + str(report))

            self.log_outputs(
                report, matrix, debug, scores, output_path / Path(name).stem
            )

            name2report[name] = report

        avg_report = self.average_report(*list(name2report.values()))
        avg_report.to_csv(Path(output_path) / "summary.csv")

        logger.info("Average report:\n" + str(avg_report))

        return name2report
