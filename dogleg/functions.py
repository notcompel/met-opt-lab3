import numpy as np


def f1(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def g1(x):
    return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


def h1(x):
    return np.array([
        [-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]],
        [-400 * x[0], 200]]
    )


x1s = [np.array([2, 1]), np.array([0, 1]), np.array([-1, 1])]


def a2(x):
    return np.exp(-1. / (100 * (x[0] - 1)) ** 2)


def f2(x):
    return x[0] ** 2 + a2(x) - 1


def g2(x):
    return np.array([2 * x[0] + a2(x) * 200 * (1. / (100 * (x[0] - 1)) ** 3)])


def h2(x):
    return np.array([2 + a2(x) * ((200 * (1. / (100 * (x[0] - 1)) ** 3)) ** 2 + 600 * (1. / (100 * (x[0] - 1)) ** 4))])


x2s = [np.array([-1]), np.array([-2])]
