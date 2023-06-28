import numpy as np


def func_normalization(f, s):
    return lambda x: f(x * np.asarray([s, 1]))
