import csv
from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class Report:
    def __init__(self, dataset, threshold=0.5):
        self.threshold = threshold
        detected_df = dataset.prediction_df.applymap(lambda x: x >= self.threshold)

        flat_result = detected_df.stack().values
        ground_truth = dataset.groundtruth_df.stack().values

        for col in dataset.groundtruth_df.columns:
            logger.info(f"Column {col} classification report:")
            logger.info(classification_report(dataset.groundtruth_df[col].values, detected_df[col].values))

        self.report = pd.DataFrame(
            classification_report(ground_truth, flat_result, output_dict=True)
        ).transpose()



        self.matrix = pd.DataFrame(
            confusion_matrix(ground_truth, flat_result, labels=[True, False]),
            columns=["True", "False"],
        )

        self.debug = self.debug(
            dataset, detected_df
        )

    def debug(self, dataset, result_df):
        def get_prob(x):
            return dataset.prediction_df.loc[x["id"], x["col"]]

    
        debug_df = pd.DataFrame()
        score_sr = dataset.dirty_df.stack()
        debug_df["from"] = score_sr.values.tolist()
        debug_df["to"] = dataset.clean_df.stack().values.tolist()
        debug_df.index = score_sr.index
        debug_df.index.names = ["id", "col"]

        debug_df["id"] = debug_df.index.get_level_values("id")
        debug_df["col"] = debug_df.index.get_level_values("col")
        debug_df["score"] = (dataset.prediction_df.stack().values > 0.5).tolist()
        debug_df["result"] = dataset.groundtruth_df.stack().values.tolist()
        debug_df["compare"] = debug_df["from"] == debug_df["to"]
        return debug_df

    def serialize(self, output_path):
        report_path = Path(output_path) / "report.csv"
        debug_path = Path(output_path) / "debug.csv"
        matrix_path = Path(output_path) / "matrix.csv"

        output_path.mkdir(parents=True, exist_ok=True)

        self.report["index"] = self.report.index
        self.report.to_csv(report_path, index=False, quoting=csv.QUOTE_ALL)

        self.debug.to_csv(debug_path, index=False, quoting=csv.QUOTE_ALL)

        self.matrix["index"] = pd.Series(["True", "False"])
        self.matrix.to_csv(matrix_path, index=None, quoting=csv.QUOTE_ALL)