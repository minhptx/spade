def jaccard_sim(x, y):
    x = set(x)
    y = set(y)

    return len(x.intersection(y)) * 1.0 / len(x.union(y))
