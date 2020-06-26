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

        with open(self.debug_file, "w") as f:
            writer = csv.writer(f)
            for tup in zip(raw_strings, cleaned_strings, predictions):
                writer.writerow(
                    [tup[1]] + list(tup[2]) + [(tup[0] != tup[1]) and tup[2][2] == False]
                )

        for idx, r_string in enumerate(raw_strings):
            if r_string != cleaned_strings[idx]:
                groundtruth.append(True)
            else:
                groundtruth.append(False)

        print(classification_report(groundtruth, [x[2] for x in predictions]))
