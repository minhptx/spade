import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rapidjson as json
from gensim.models import Doc2Vec

from utils.features import (
    extract_bag_of_characters_features,
    extract_word_embeddings_features,
    extract_bag_of_words_features,
)
from utils.features.bag_of_regexes import extract_bag_of_regexes
from utils.features.paragraph_vector import infer_paragraph_embeddings_features


def prepare_embedding_vectors(embedding_path):
    word_vectors_f = open(embedding_path, encoding="utf-8")
    word_to_embedding = {}

    for w in word_vectors_f:
        term, vector = w.strip().split(" ", 1)
        vector = np.array(vector.split(" "), dtype=float)
        word_to_embedding[term] = vector
    return word_to_embedding


def build_features(df, model_path, start_index=None, end_index=None):

    feature_df = pd.DataFrame()
    word_to_embedding = prepare_embedding_vectors(
        os.path.join(model_path, "glove.6B.50d.txt")
    )

    vec_dim = 400
    model = Doc2Vec.load(os.path.join(model_path, "par_vec_trained_400.pkl"))

    n_samples = 1000
    i = 0

    if end_index is not None:
        df = df.head(end_index)

    if start_index is not None:
        df = df.iloc[start_index:]

    for idx, column in df.iterrows():
        column_to_value = {"title": column["title"], "source": column["source"]}
        raw_sample = column["data"]

        i = i + 1
        if i % 20 == 0:
            print("Extracting features for data column ", i)

        n_values = len(raw_sample)

        if n_samples > n_values:
            n_samples = n_values

        # Sample n_samples from data column, and convert cell values to string values
        raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

        column_to_value["BOC"] = extract_bag_of_characters_features(raw_sample)

        column_to_value["BOW"] = extract_word_embeddings_features(
            raw_sample, word_to_embedding
        )

        column_to_value["EMB"] = extract_bag_of_words_features(raw_sample, n_samples)

        column_to_value["PARA"] = infer_paragraph_embeddings_features(
            raw_sample, model, vec_dim
        )

        column_to_value["REGEX"] = extract_bag_of_regexes(raw_sample)

        feature_df = feature_df.append(column_to_value, ignore_index=True)

    feature_df.fillna(feature_df.mean(), inplace=True)

    return feature_df


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]
    e_path = sys.argv[2]
    o_path = sys.argv[3]

    data_df = pd.read_csv(input_path)
    feature_df = build_features(data_df, model_path=e_path)

    json_struct = json.loads(feature_df.to_json(orient="records"))
    df_flat = pd.io.json.json_normalize(json_struct)

    df_flat.to_csv(Path(o_path) / f"{Path(input_path).stem}.csv", index=False)
