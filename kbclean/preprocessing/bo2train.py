from pathlib import Path
import pandas as pd
import random
import numpy as np

if __name__ == "__main__":
    data_path = Path("data/raw/bo")
    output_path = Path("data/test/bo")

    cleaned_path = output_path / "cleaned"
    raw_path = output_path / "raw"

    cleaned_path.mkdir(parents=True, exist_ok=True)
    raw_path.mkdir(parents=True, exist_ok=True)

    for file_path in data_path.iterdir():
        df = pd.read_csv(file_path, keep_default_na=False)
        for i in range(1):
            cleaned_data = df.iloc[:, 1].values.tolist()
            raw_data = df.iloc[:, 0].values.tolist()

            num_cleaned = random.randint(int(0.7 * len(raw_data)), int(0.95 * len(raw_data)))
            indices = np.random.choice(len(raw_data), num_cleaned, replace=False)

            for index in indices:
                raw_data[index] = cleaned_data[index]

            pd.DataFrame(cleaned_data, columns=["data"]).to_csv(cleaned_path / f"{file_path.stem}_{i}.csv", index=False)
            pd.DataFrame(raw_data, columns=["data"]).to_csv(raw_path / f"{file_path.stem}_{i}.csv", index=False)
        
