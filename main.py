import math

import numpy as np
from numpy import cos, sin

from bfgs import BFGS
from lbfgs import LBFGS
from descent import GradientDescent
from linear_regression import test_linear_regression, generate_functions_and_grads


def const_lr(step):
    return 0.01


def f(x):
    # return x[0] ** 2 - x[0] * x[1] + x[1] ** 2 + 9 * x[0] - 6 * x[1] + 20
    return 10 * x[0] ** 2 + x[1] ** 2
    # return x[0] ** 4 + x[1] ** 4 - 4 * x[0] ** 2 - 5 * x[1] ** 2 - x[0] - x[1]


def grad_f(x):
    # return np.array([2 * x[0] - x[1] + 9, -x[0] + 2 * x[1] - 6])
    return np.array([20 * x[0], 2 * x[1]])
    # return np.array([4 * x[0] ** 3 - 8 * x[0] - 1, 4 * x[1] ** 3 - 10 * x[1] - 1])


# Testing
points, f_set, grad_set = generate_functions_and_grads(50)

test_f = f
test_grad = grad_f

descent = LBFGS(
    test_f, test_grad, np.array([3.0, -3.0]), const_lr
)

descent.process()

