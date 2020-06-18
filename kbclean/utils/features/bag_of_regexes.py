from collections import OrderedDict

import numpy as np
import pandas as pd
import regex as re
from scipy.stats import kurtosis, skew


def to_regex(string1):
    string1 = re.sub("[A-Z]", "A", string1)
    string1 = re.sub("[a-z]", "a", string1)
    string1 = re.sub("[0-9]", "0", string1)
    return string1


def extract_bag_of_regexes(data):

    f = {}

    data = pd.Series(data)

    data_no_null = data.dropna().apply(lambda x: to_regex(x))
    all_value_features = OrderedDict()

    for c in ["[A]", "[a]", "[0]"]:
        all_value_features["n_{}".format(c)] = data_no_null.str.count(c)

    for value_feature_name, value_features in all_value_features.items():
        f["{}-agg-any".format(value_feature_name)] = float(any(value_features))
        f["{}-agg-all".format(value_feature_name)] = float(all(value_features))
        f["{}-agg-mean".format(value_feature_name)] = np.mean(value_features)
        f["{}-agg-var".format(value_feature_name)] = np.var(value_features)
        f["{}-agg-min".format(value_feature_name)] = np.min(value_features)
        f["{}-agg-max".format(value_feature_name)] = np.max(value_features)
        f["{}-agg-median".format(value_feature_name)] = np.median(value_features)
        f["{}-agg-sum".format(value_feature_name)] = np.sum(value_features)
        f["{}-agg-kurtosis".format(value_feature_name)] = kurtosis(value_features)
        f["{}-agg-skewness".format(value_feature_name)] = skew(value_features)

    return f
