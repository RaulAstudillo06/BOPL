# Copyright (c) 2019, Raul Astudillo Marban

import numpy as np


def random_preference_elicitation(Y):
    n_attributes = len(Y)
    suggested_pair = []
    indices = np.random.choice(len(Y[0]), 2, replace=False)
    for i in indices:
        suggested_pair.append(np.atleast_1d([Y[j][i, 0] for j in range(n_attributes)]))
    return suggested_pair
