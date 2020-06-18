import torch


def exponent_neg_manhattan_distance(x1, x2):
    return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1, keepdim=True))
