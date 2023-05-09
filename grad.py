import numpy as np


def taylor_grad(f: callable, x) -> callable:
    eps = 0.00001

    result = np.array([0.0] * len(x))
    for i in range(len(x)):
        h = np.array([0.0] * len(x))
        h[i] = eps
        result[i] = (f(x + h) - f(x - h)) / (eps * 2)
    return result
