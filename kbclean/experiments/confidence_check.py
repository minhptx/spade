from kbclean.datasets.dataset import Dataset
from pathlib import Path
import pandas as pd
from kbclean.recommendation.psl import PSLearner
from kbclean.utils.inout import load_config
import numpy as np

if __name__ == "__main__":
    path = Path("data/test/data_gov")
    output_path = Path("data/test/label_gov")
    output_path.mkdir(parents=True, exist_ok=True)
    example_path = Path("data/test/example_gov")
    example_path.mkdir(parents=True, exist_ok=True)
    hparams = load_config("config/lstm")
    hparams.debug_dir = f"debug/{path.name}"
    Path(hparams.debug_dir).mkdir(parents=True, exist_ok=True)
    top_columns = []
    top_scores = []
    top_examples = []
    for file_path in path.iterdir():
        print(hparams.debug_dir)
        try:
            dirty_df = pd.read_csv(
                file_path, header=0, dtype=str, keep_default_na=False
            )
            dirty_df.columns = [x.replace("/", "_") for x in dirty_df.columns]
        except:
            print(f"Error in file {file_path}")
            continue

        if len(dirty_df) > 100000:
            continue
        dataset = Dataset(dirty_df, None)

        dirty_df.columns = [x.replace("/", "_") for x in dataset.dirty_df.columns]
        recommender = PSLearner(dataset, hparams)

        try:
            for col in dataset.dirty_df.columns:

                score, examples = recommender.fit_score(col, 20)

                top_scores.append(score)
                top_columns.append(dataset.dirty_df[col].values.tolist())
                top_examples.append(examples)
        except:
            continue

    top_100_indices = np.random.choice(
        range(len(top_scores)), p=np.asarray(top_scores) / sum(top_scores)
    )
    top_scores = 1 - np.asarray(top_scores)

    bottom_100_indices = np.random.choice(
        range(len(top_scores)), p=np.asarray(top_scores) / sum(top_scores)
    )
    for index in top_100_indices.tolist() + bottom_100_indices.tolist():
        pd.DataFrame(top_columns[index], columns=["data"]).to_csv(
            (output_path / f"{index}.csv"), index=None
        )
        pd.DataFrame(top_examples[index], columns=["data"]).to_csv(
            (example_path / f"{index}.csv"), index=None
        )
