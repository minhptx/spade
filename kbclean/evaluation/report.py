import csv

from sklearn.metrics import classification_report


class Report:
    def __init__(self, debug_path):
        self.debug_path = debug_path
        self.groundtruth = []
        self.predictions = []     

    def clear(self):
        self.groundtruth.clear()
        self.predictions.clear()   

    def __call__(self, raw_strings, cleaned_strings, predictions):
        assert len(raw_strings) == len(
            predictions
        ), "Input and target data should have the same size"

        groundtruth = []

        for idx, r_string in enumerate(raw_strings):
            if r_string != cleaned_strings[idx]:
                groundtruth.append(True)
            else:
                groundtruth.append(False)

        self.groundtruth.extend(groundtruth)
        self.predictions.extend(predictions)

        with open(self.debug_file, "a") as f:
            writer = csv.writer(f)
            for raw_string, cleaned_string, gt, prediction in zip(
                raw_strings, cleaned_strings, groundtruth, predictions
            ):
                if gt != prediction:
                    writer.writerow([cleaned_string, raw_string, gt, prediction])

    def publish(self):
        return classification_report(self.groundtruth, self.predictions)
