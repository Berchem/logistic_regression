import numpy as np

TOL = 1e8
LIM = 100


def sigmoid(array_like):
    return 1 / (1 + np.exp(-array_like))


def _add_intercept(array_like):
    array_like