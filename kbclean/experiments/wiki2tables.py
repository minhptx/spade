import pandas as pd
import sys
from pathlib import Path

from pandas.tseries import offsets
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    used_urls = []
    index = 0

    len_values = []

    for file_path in path.iterdir():
        try:
            print(file_path)
            df = pd.read_csv(file_path, delimiter="\t", header=None, names=["url", "caption", "section", "column_id", "column_header", "values", "error", "score", "label"], keep_default_na=None)

            for _, row in df.iterrows():
                if row["url"] in used_urls:
                    continue
                else:
                    used_urls.append(row["url"])
                values = row["values"].split("___")
                len_values.append(len(values))
                errors = row["error"].split("---")
                new_df = pd.DataFrame({"value": values})

                new_df.to_csv(output_path / f"{str(index)}.csv", index=None)
                index += 1
        except Exception as e:
            print(e)
            continue

    n, bins, patches = plt.hist(x=len_values, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Size')
    plt.ylabel('Number of tables')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig("histogram.png")