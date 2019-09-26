# Copyright (c) 2019, Raul Astudillo

import numpy as np


def preference_encoder(a, b):
    a_aux = np.squeeze(a)
    b_aux = np.squeeze(b)
    if a_aux > b_aux:
        preference = 1
    elif a_aux < b_aux:
        preference = -1
    else:
        preference = 0
    return preference
