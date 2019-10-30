import numpy as np

def chebyshev_scalarization(Y, weight):
    tmp = np.multiply(weight, Y.T).T
    return np.min(tmp, axis=0) + 0.05 * np.sum(Y, axis=0)