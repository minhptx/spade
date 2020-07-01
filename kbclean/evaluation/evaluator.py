import csv

from sklearn.metrics import classification_report


class Evaluator:
    def __init__(self, debug_file):
        self.debug_file = debug_file

    def evaluate(self, raw_strings, cleaned_strings, predictions):
        assert len(raw_strings) == len(
            predictions
        ), "Input and target data should have the same size"

        groundtruth = []

        for idx, r_string in enumerate(raw_strings):
            if r_string != cleaned_strings[idx]:
                groundtruth.append(True)
            else:
                groundtruth.append(False)

        with open(self.debug_file, "w") as f:
            writer = csv.writer(f)
            for raw_string, cleaned_string, gt, prediction in zip(
                raw_strings, cleaned_strings, groundtruth, predictions
            ):
                if gt != prediction:
                    writer.writerow(
                        [cleaned_string, raw_string, groundtruth, prediction]
                    )

        print(classification_report(groundtruth, predictions))
