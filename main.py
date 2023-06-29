import numpy as np
from matplotlib import pyplot as plt

from bfgs import BFGS
from descent import GradientDescent
from lbfgs import LBFGS
from linear_regression import generate_functions_and_grads
from newton import gauss_newton
from upgrades import UpgradeType


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
# points, f_set, grad_set = generate_functions_and_grads(50)
#
# test_f = f
# test_grad = grad_f
#
# bfgs = BFGS(
#     test_f, test_grad, np.array([3.0, -3.0])
# )
# bfgs.process()
#
# lbfgs = LBFGS(
#     test_f, test_grad, np.array([3.0, -3.0])
# )
# lbfgs.process()


# test_f = f
# test_grad = grad_f
#
# descent = GradientDescent(
#     test_f, test_grad, np.array([3.0, -3.0]), const_lr
# )


# descent.process(UpgradeType.Empty)
# descent.process(UpgradeType.Nesterov)
# descent.process(UpgradeType.Momentum)
# descent.process(UpgradeType.AdaGrad)
# descent.process(UpgradeType.RMSProp)
# descent.process(UpgradeType.Adam)

def f1(x, ab):
    a, b = ab
    return a * np.exp(-b * x)


def f2(x, ab):
    a, b = ab
    return a * np.sin(-b * x)


def f3(x, ab):
    a, b = ab
    return a * np.exp(-b * x) + a * np.sin(-b * x)


def f4(x, ab):
    a, b = ab
    return np.exp(-a * x) + b * x


def private_residual(v, f, x_data, y_data):
    return y_data - f(x_data, v)


def jacobian(v, residual):
    eps = 1e-6
    real = residual(v)

    J = []
    for i in range(len(v)):
        v_shifted = v.copy()
        v_shifted[i] += eps
        shifted = residual(v_shifted)

        J.append((shifted - real) / eps)
    J = np.array(J).T

# bfgs = BFGS(
#     test_f, test_grad, np.array([3.0, -3.0]), const_lr
# )
# bfgs.process()
#
#
# lbfgs = LBFGS(
#     test_f, test_grad, np.array([3.0, -3.0]), const_lr
# )
# lbfgs.process()
#
#     x_data = np.linspace(0, 1, 20)
#     ab_true = np.array([1.0, 0.1])
#     y_data = f(x_data, ab_true)
#
#     ab_start = np.array([10.0, 5.0])
#
# descent = GradientDescent(
#     test_f, test_grad, np.array([3.0, -3.0]), const_lr
# )


# descent.process(UpgradeType.Empty)
# descent.process(UpgradeType.Nesterov)
# descent.process(UpgradeType.Momentum)
# descent.process(UpgradeType.AdaGrad)
# descent.process(UpgradeType.RMSProp)
# descent.process(UpgradeType.Adam)

