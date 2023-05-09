import random

import numpy as np

import matplotlib.pyplot as plt

from descent import end_condition, jump_condition, log_steps
from upgrades import UpgradeCore


# Linear Regression

def random_float(_from: float, _to: float) -> float:
    return _from + (_to - _from) * random.random()


# Generates point (x, y) near Y = Kx + B line.
# Returns x, y, function f(k, b) = (y - kx - b)^2
# and grad(k, b) = [-2x(y-kx-b), -2(y-kx-b)].
def generate_point_function(K: float, B: float) -> [float, float, callable, callable]:
    x = random_float(0, 10)
    y = (K * x + B) + random_float(-3.0, 3.0)

    def _f(kb):
        return (y - kb[0] * x - kb[1]) ** 2

    def _grad(kb):
        return np.array([-2.0 * x * (y - kb[0] * x - kb[1]), -2.0 * (y - kb[0] * x - kb[1])])

    return x, y, _f, _grad


# Returns [[x1, y1], [x2, y2], ..., [xn, yn]] [f1, f2, ..., fn], [grad1, grad2, ..., grad_n]
def generate_functions_and_grads(n: int) -> [list[callable], list[callable]]:
    points, f_set, grad_set = [], [], []
    k, b = random_float(-2.0, 2.0), random_float(1.0, 3.0)

    for i in range(n):
        x, y, f, grad = generate_point_function(k, b)

        points.append(np.array([x, y]))
        f_set.append(f)
        grad_set.append(grad)

    return points, f_set, grad_set


def calculate_batch(grad_set, batch_size, iteration, kb, lr, upgrade_core: UpgradeCore):
    result = np.array([0.0, 0.0])
    for i in range(batch_size):
        k = (iteration * batch_size + i) % len(grad_set)
        result += upgrade_core.calculate_grad(grad_set[k], kb, iteration, lr)
    return result / batch_size


def test_linear_regression(
        point_number: int,
        batch_size: int,
        learning_rate: callable,
        upgrade_type,
        points, f_set, grad_set
):
    upgrade_core = UpgradeCore(upgrade_type, 2)

    # points, f_set, grad_set = generate_functions_and_grads(point_number)

    vector = np.array([-3.0, -3.0])
    hist = [tuple(vector)]

    eps = 0.001
    it = 0

    while it < 100000 and end_condition(hist, eps, vector):
        if it > 10 and jump_condition(hist, eps, vector):
            print("Спуск не сошелся.")
            break

        lr = learning_rate(it)
        grad_v = calculate_batch(grad_set, batch_size, it, vector, lr, upgrade_core)
        vector -= lr * grad_v

        hist.append(tuple(vector))
        # if it % 10 == 0:
        #     log_steps(it, hist, vector)
        it += 1

    print(it)

    def f(w):
        return sum([f_part(w) for f_part in f_set])

    print(f'Loss function: {f(hist[-1])}')

    if (batch_size - 1) % 10 == 0:
        visualize_f_set(f_set, hist)
        visualize_line(vector[0], vector[1], points, learning_rate.__name__)


def visualize_f_set(f_set, hist):
    def f(w):
        return sum([f_part(w) for f_part in f_set])

    # print(f'Loss function: {f(hist[-1])}')

    # Define the range of values for x and y
    x_vals = np.linspace(-3, 3, 1000)
    y_vals = np.linspace(-3, 3, 1000)

    # Create a 2D meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute the function values over the meshgrid
    Z = np.squeeze(f([X, Y]))

    # Plot the level lines of the function
    plt.contour(X, Y, Z, levels=60)

    # Plot the path taken by gradient descent
    plt.plot(*zip(*hist), 'o-', color='r')

    # Show the plot
    plt.show()


def visualize_line(k, b, points, name="name"):
    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    plt.plot(x_values, y_values, 'ro')

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = k * x_min + b, k * x_max + b
    plt.plot([x_min, x_max], [y_min, y_max], 'b-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{name}: y = {round(k, 3)}x + {round(b, 3)}')
    plt.show()
