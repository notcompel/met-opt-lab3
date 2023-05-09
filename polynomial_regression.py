import math
import random
from enum import Enum

import numpy as np

import matplotlib.pyplot as plt
from descent import end_condition, jump_condition, log_steps
from linear_regression import random_float, visualize_f_set
from upgrades import UpgradeCore


class Mode(Enum):
    Simple = 0
    L1 = 1
    L2 = 2
    Elastic = 3


def gamma():
    return 0.5


def simple_f(w):
    return 0.0


def simple_grad_f(w):
    return 0.0


def l1_f(w):
    return gamma() * sum([abs(wi) for wi in w])


def l1_grad_f(w):
    return gamma() * np.array([math.copysign(1, wi) for wi in w])


def l2_f(w):
    return gamma() * sum([wi ** 2 for wi in w])


def l2_grad_f(w):
    return gamma() * 2.0 * w


def elastic_f(w):
    return l1_f(w) + l2_f(w)


def elastic_grad_f(w):
    return l1_grad_f(w) + l2_grad_f(w)


# Polynomial Regression

# Generates point (x, y) near Y = W0x^0 + W1x^1 + W2x^2 + ... + Wnx^n line.
# Returns x, y, function f([w0, ..., wn]) = ((k=0..n)∑(wk * x^k) - y)^2
# and grad([w0, ..., wn]) = [ (i=0..n) | 2x^i( (k=0..n)∑(wk * x^k) - y) ].
def generate_point_function_poly(W, mode=Mode.Simple) -> [float, float, callable, callable]:
    mode_map = {
        Mode.Simple: (simple_f, simple_grad_f),
        Mode.L1: (l1_f, l1_grad_f),
        Mode.L2: (l2_f, l2_grad_f),
        Mode.Elastic: (elastic_f, elastic_grad_f),
    }

    df, dgrad = mode_map.get(mode)

    x = random_float(-2.0, 2.0)
    x_poly = np.power(x, np.arange(len(W)))

    y = random_float(-2.0, 2.0)

    # y = np.dot(x_poly, W) + random_float(-1.0, 1.0)
    # y = np.dot(x_poly, W)

    def _f(w):
        return (np.dot(x_poly, w) - y) ** 2 + df(w)

    def _grad(w):
        return x_poly * 2.0 * (np.dot(x_poly, w) - y) + dgrad(w)

    return x, y, _f, _grad


# Returns [[x1, y1], [x2, y2], ..., [xn, yn]] [f1, f2, ..., fn], [grad1, grad2, ..., grad_n]
def generate_poly_functions_and_grads(point_number: int, n: int, mode=Mode.Simple) -> [list[callable], list[callable]]:
    points, f_set, grad_set = [], [], []
    W = np.array([random_float(-2, 2) / math.pow(10.0, i) for i in range(n + 1)])
    # W = np.array([0.0, 9.0, -6.0, 3.0, -0.5])

    for i in range(point_number):
        x, y, f, grad = generate_point_function_poly(W, mode)

        points.append(np.array([x, y]))
        f_set.append(f)
        grad_set.append(grad)

    return points, f_set, grad_set


def calculate_poly_batch(grad_set, batch_size, iteration, vector, lr, upgrade_core: UpgradeCore):
    result = np.array([0.0] * len(vector))
    for i in range(batch_size):
        k = (iteration * batch_size + i) % len(grad_set)
        result += upgrade_core.calculate_grad(grad_set[k], vector, iteration, lr)
    return result / batch_size


def test_polynomial_regression(
        points_number: int,
        n: int,
        batch_size: int,
        learning_rate: callable,
        mode=Mode.Simple
):
    upgrade_core = UpgradeCore(n + 1)

    points, f_set, grad_set = generate_poly_functions_and_grads(points_number, n, mode)

    # vector = np.array([random_float(-0.2, 0.2) / math.pow(10, i) for i in range(n + 1)])
    vector = np.array([0.0] * (n + 1))
    hist = [tuple(vector)]

    eps = 0.0000001
    it = 0

    while it < 150000 and end_condition(hist, eps, vector):
        # if it > 2 and jump_condition(hist, eps, vector):
        #     print("Спуск не сошелся.")
        #     break

        lr = learning_rate(it)
        grad_v = calculate_poly_batch(grad_set, batch_size, it, vector, lr, upgrade_core)
        vector -= lr * grad_v

        hist.append(tuple(vector))
        if it % 10 == 0:
            log_steps(it, hist, vector)
        it += 1

    print("Finished regression. {} iterations total.".format(it))
    visualize_curve(vector, points, mode.name)

    return


def visualize_curve(w, points, name="name"):
    def _f(_x):
        return np.dot(w, np.power(_x, np.arange(len(w))))

    x = np.linspace(min(p[0] for p in points), max(p[0] for p in points), 100)  # adjust range as needed
    y = np.array([_f(xi) for xi in x])

    # Plot the points using matplotlib.pyplot.scatter()
    plt.scatter([p[0] for p in points], [p[1] for p in points], c='red', label='Data Points')

    # Plot the computed y values using matplotlib.pyplot.plot()
    plt.plot(x, y, c='blue', label='Fitted Function')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'n = {len(w)}, MODE: {name}')

    plt.legend()
    plt.show()
