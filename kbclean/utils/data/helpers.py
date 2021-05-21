from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import regex as re
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import Vocab


def not_equal(df1, df2):
    return (df1 != df2) & ~(df1.isnull() & df2.isnull())


def diff_dfs(df1, df2, compare_func=not_equal):
    assert (df1.columns == df2.columns).all(), "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        diff_mask = compare_func(df1, df2)
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ["id", "col"]
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        df = pd.DataFrame({"from": changed_from, "to": changed_to}, index=changed.index)
        df["id"] = df.index.get_level_values("id")
        df["col"] = df.index.get_level_values("col")
        return df


def zip_dfs(df1, df2):
    stacked_df1 = df1.stack()
    stacked_df1.index.names = ["id", "col"]
    stacked_df2 = df2.stack()
    df = pd.DataFrame({"from": stacked_df1, "to": stacked_df2}, index=stacked_df1.index)
    df["id"] = df.index.get_level_values("id")
    df["col"] = df.index.get_level_values("col")
    return df


def str2regex(x, match_whole_token=True):
    if not match_whole_token:
        try:
            if x is None:
                return ""
            x = re.sub(r"[A-Z]", "A", x)
            x = re.sub(r"[0-9]", "0", x)
            x = re.sub(r"[a-z]", "a", x)
            return x
        except Exception as e:
            print(e, x)
            return x
    try:
        if x is None:
            return ""
        x = re.sub(r"[A-Z]+", "A", x)
        x = re.sub(r"[0-9]+", "0", x)
        x = re.sub(r"[a-z]+", "a", x)
        x = re.sub(r"Aa", "C", x)
        return x
    except Exception as e:
        print(e, x)
        return x


def unzip_and_stack_tensors(tensor):
    transpose_tensors = list(zip(*tensor))
    result = []
    for tensor in transpose_tensors:
        result.append(torch.stack(tensor, dim=0))
    return result


def split_train_test_dls(
    data, collate_fn, batch_size, ratios=[0.7, 0.2], pin_memory=True, num_workers=16
):
    train_length = int(len(data) * ratios[0])
    val_length = int(len(data) * ratios[1])
    train_dataset, val_dataset, test_dataset = random_split(
        data, [train_length, val_length, len(data) - train_length - val_length],
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_dataset,  
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_dl, val_dl, test_dl


def build_vocab(flatten_data):
    counter = Counter(flatten_data)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    return Vocab(ordered_dict)
