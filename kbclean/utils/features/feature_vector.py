import random
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import rapidjson as json

from utils.features import (
    extract_bag_of_characters_features,
    extract_word_embeddings_features,
    extract_bag_of_words_features,
)


def prepare_embedding_vectors(embedding_path):
    word_vectors_f = open(embedding_path, encoding="utf-8")
    word_to_embedding = {}

    for w in word_vectors_f:

        term, vector = w.strip().split(" ", 1)
        vector = np.array(vector.split(" "), dtype=float)
        word_to_embedding[term] = vector
    return word_to_embedding


def build_features(data, embedding_path, output_path):
    output_path = Path(output_path)

    df_char = pd.DataFrame()
    df_word = pd.DataFrame()
    df_word_embedding = pd.DataFrame()
    word_to_embedding = prepare_embedding_vectors(embedding_path)

    n_samples = 1000
    i = 0
    for idx, raw_sample in data.iterrows():
        title, data = raw_sample["title"], raw_sample["data"]
        i = i + 1
        if i % 20 == 0:
            print("Extracting features for data column ", i)

        n_values = len(raw_sample)

        if n_samples > n_values:
            n_samples = n_values

        extract_word_embeddings_features1 = partial(
            extract_word_embeddings_features, word_to_embedding=word_to_embedding
        )

        # Sample n_samples from data column, and convert cell values to string values
        raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

        df_char = df_char.append(extract_bag_of_characters_features(raw_sample), ignore_index=True)
        df_word = df_word.append(extract_word_embeddings_features1(raw_sample), ignore_index=True)
        df_word_embedding = df_word_embedding.append(extract_bag_of_words_features(raw_sample), ignore_index=True)

    df_char.fillna(df_char.mean(), inplace=True)
    df_word.fillna(df_word.mean(), inplace=True)
    df_word_embedding.fillna(df_word_embedding.mean(), inplace=True)

    print(len(df_char))
    df_char.to_csv(output_path / "char.csv")
    df_word.to_csv(output_path / "word.csv")
    df_word_embedding.to_csv(output_path / "embedding.csv")


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]
    e_path = sys.argv[2]
    o_path = sys.argv[3]
    columns = json.load(open(input_path, "r"))

    df = pd.DataFrame(columns, columns=["title", "data"], dtype=object)
    build_features(df, embedding_path=e_path, output_path=o_path)
