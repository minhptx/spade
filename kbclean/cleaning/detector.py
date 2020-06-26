from typing import List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class Detector:
    def __init__(
        self, hparams, lm_model, outlier_method="SVM", error_method="",
    ):
        self.lm_model = lm_model

        self.outlier_method = outlier_method
        self.error_method = error_method

        self.hparams = hparams

        if self.outlier_method == "RF":
            self.outlier_model = IsolationForest()
        elif self.outlier_method == "SVM":
            self.outlier_model = OneClassSVM()

    def detect_outliers(self, strings: List[str]):
        tensors = []

        for i in range(0, len(strings), self.hparams.batch_size):
            self.lm_model.predict(strings[i : i + self.hparams.batch_size])
            print(tensor)
            tensors.append(tensor)

        outputs = self.outlier_model.fit_predict((np.concatenate(tensors, axis=0)))

        return list(map(lambda x: x[1] == -1, zip(strings, outputs)))

    def detect_errors(self, strings: List[str]):
        tensors = []

        for i in range(0, len(strings), self.hparams.batch_size):
            tensor = (
                self.lm_model.predict(strings[i : i + self.hparams.batch_size])
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )
            tensors.append(tensor)

        probs = np.concatenate(tensors, axis=0).reshape(-1, 1)
        preds = self.outlier_model.fit_predict(probs)

        return zip(strings, probs[:, 0], preds == -1)

    def detect(self, strings: List[str]):
        # errors = self.detect_outliers(strings)
        errors = self.detect_errors(strings)

        return list(errors)

        # return list(map(lambda x: x[0] or x[1], zip(outliers, errors)))
