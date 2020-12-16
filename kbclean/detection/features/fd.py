from collections import defaultdict
from kbclean.detection.features.base import BaseFeaturizer
import torch


from kbclean.datasets.dataset import Dataset
import numpy as np

class FDFeaturizer(BaseFeaturizer):
    def fit(self, dataset: Dataset, col: str):
        pass

    def transform(self, dataset: Dataset, col: str):
        mapping_arrs = []
        for col1 in dataset.dirty_df.columns.values:
            if col1 == col:
                continue
            mappings = defaultdict(list)
            mapping_dict = defaultdict(dict)

            lhs_values = dataset.dirty_df[col1].values.tolist()
            rhs_values = dataset.dirty_df[col].values.tolist()

            for index, l_value in enumerate(lhs_values):
                mappings[l_value].append(rhs_values[index])

            for l_value in mappings.keys():
                for r_value in mappings[l_value]:
                    if len(mappings[l_value]) > 1:
                        mapping_dict[l_value][r_value] = 0
                    else:
                        mapping_dict[l_value][r_value] = 1
            
            mapping_arrs.append(torch.tensor([mapping_dict[l_value][r_value] for l_value, r_value in zip(lhs_values, rhs_values)]).view(-1, 1))

        return [torch.cat(mapping_arrs, dim=1)]


    def n_features(self, dataset: Dataset):
        return len(dataset.dirty_df.columns) - 1

    def feature_dim(self):
        return 1