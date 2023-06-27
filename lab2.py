import math
import os
from copy import copy
from enum import Enum
import random

import psutil
import time
import numpy as np
import matplotlib.pyplot as plt


# Measurement

def measure_memory_usage(func):
    """Decorator function to measure memory usage of a function"""

    def wrapper(*args, **kwargs):
        # Start process memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024.0 / 1024.0
        # Call the function and get the result
        result = func(*args, **kwargs)
        # End process memory usage
        end_memory = process.memory_info().rss / 1024.0 / 1024.0
        # Calculate memory usage difference
        mem_diff = end_memory - start_memory
        print(f"Memory used: {mem_diff:.2f} MB")
        return result

    return wrapper


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.5f} seconds to execute.")
        return result

    return wrapper


# Visualization

def visualize(f, hist):
    # Define the range of values for x and y
    x_vals = np.linspace(-10, 10, 1000)
    y_vals = np.linspace(-10, 10, 1000)

    # Create a 2D meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute the function values over the meshgrid
    Z = np.squeeze(f([X, Y]))

    # Plot the level lines of the function
    plt.contour(X, Y, Z, levels=30)

    # Plot the path taken by gradient descent
    plt.plot(*zip(*hist), 'o-', color='r')

    # Show the plot
    plt.show()


# Upgrades

class UpgradeType(Enum):
    Empty = 0,
    Momentum = 1,
    Nesterov = 2,
    AdaGrad = 3,
    RMSProp = 4,
    Adam = 5


def nothing(g_i, it):
    return g_i


class UpgradeCore:
    def __init__(self, upgrade_type: UpgradeType, dim):
        self.eps = 0.000001

        self.upgrade_type = upgrade_type

        self.v = np.array([[0.0] * dim] * 200000)
        self.G = [np.array([0.0000001] * dim)]

        grad_map = {
            UpgradeType.Empty: nothing,
            UpgradeType.Nesterov: self.momentum,
            UpgradeType.Momentum: self.momentum,
            UpgradeType.AdaGrad: self.adagrad,
            UpgradeType.RMSProp: self.rms_prop,
            UpgradeType.Adam: self.adam
        }

        self.main_method = grad_map.get(self.upgrade_type)

    def calculate_grad(self, grad_f, vector, it, lr):
        if self.upgrade_type == UpgradeType.Nesterov:
            grad_f = self.nesterov_grad_f(grad_f, it, lr)

        grad = grad_f(vector)
        return self.main_method(grad, it)

    def nesterov_grad_f(self, grad, it, lr):
        gamma = 0.8
        return lambda w: grad(w - lr * gamma * self.v[it])

    def momentum(self, g_i, it):
        gamma = 0.8
        self.v[it + 1] = gamma * self.v[it] + (1 - gamma) * g_i
        return self.v[it + 1]

    def adagrad(self, g_i, it):
        self.G.append(self.G[it] + np.square(g_i))
        return g_i / np.power(self.G[it + 1], 1 / 2)

    def rms_prop(self, g_i, it):
        gamma = 0.9
        self.G.append(gamma * self.G[it] + (1 - gamma) * np.square(g_i))
        return g_i / np.power(self.G[it + 1], 1 / 2)

    def adam(self, g_i, it):
        gamma1 = 0.9
        gamma2 = 0.999
        self.v[it + 1] = gamma1 * self.v[it] + (1 - gamma1) * g_i
        correct_v = self.v[it + 1] / (1 - gamma1 ** (it + 1))
        self.G.append(gamma2 * self.G[it] + (1 - gamma2) * np.square(g_i))
        correct_G = self.G[it + 1] / (1 - gamma2 ** (it + 1))
        return correct_v / np.power(correct_G, 1 / 2)


# Gradient Descent

class GradientDescent:
    def __init__(self, f, grad_f, start_pos, learning_rate):
        self.f = f
        self.grad_f = grad_f
        self.start_pos = start_pos
        self.learning_rate = learning_rate

    @measure_memory_usage
    @measure_time
    def process(self, upgrade_type: UpgradeType):
        core = UpgradeCore(upgrade_type, len(self.start_pos))

        print("\nStarted {}.".format(upgrade_type.name))

        hist = self.calculate_history(core)

        visualize(self.f, hist)

    def calculate_history(self, upgrade_core: UpgradeCore) -> np.array:
        vector = copy(self.start_pos)
        hist = [tuple(vector)]

        eps = 0.001
        it = 0

        while it < 10000 and end_condition(hist, eps, vector):
            if it > 10 and jump_condition(hist, eps, vector):
                print("Спуск не сошелся.")
                break

            lr = self.learning_rate(it)
            grad_v = upgrade_core.calculate_grad(self.grad_f, vector, it, lr)
            vector -= lr * grad_v

            hist.append(tuple(vector))
            if it % 10 == 0:
                log_steps(it, hist, vector)
            it += 1

        print("Finished. {} iterations total.".format(it))
        print("Minimum is in {}".format([round(x, 4) for x in hist[-1]]))
        return np.array(hist)


def end_condition(hist, eps, vector):
    return len(hist) <= 2 or any(abs(hist[-2][i] - hist[-1][i]) >= eps for i in range(len(vector)))


def jump_condition(hist, eps, vector):
    return any(all(abs(hist[-k][i] - hist[-1][i]) < eps for i in range(len(vector))) for k in range(3, 10))


def log_steps(it, hist, vector):
    print(f"{it}: {' '.join(f'%.8f' % abs(hist[-1][i]) for i in range(len(vector)))}")


# Linear regression

def random_float(_from: float, _to: float) -> float:
    return _from + (_to - _from) * random.random()


# Generates point (x, y) near Y = Kx + B line.
# Returns x, y, function f(k, b) = (y - kx - b)^2
# and grad(k, b) = [-2x(y-kx-b), -2(y-kx-b)].
def generate_point_function(K: float, B: float) -> [float, float, callable, callable]:
    x = random_float(0, 10)
    y = (K * x + B) + random_float(-1.0, 1.0)

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
        upgrade_type=UpgradeType.Momentum
):
    upgrade_core = UpgradeCore(upgrade_type, 2)

    points, f_set, grad_set = generate_functions_and_grads(point_number)

    vector = np.array([-2.0, -2.0])
    hist = [tuple(vector)]

    eps = 0.0001
    it = 0

    while it < 10000 and end_condition(hist, eps, vector):
        if it > 10 and jump_condition(hist, eps, vector):
            print("Спуск не сошелся.")
            break

        lr = learning_rate(it)
        grad_v = calculate_batch(grad_set, batch_size, it, vector, lr, upgrade_core)
        vector -= lr * grad_v

        hist.append(tuple(vector))
        if it % 10 == 0:
            log_steps(it, hist, vector)
        it += 1

    print("Finished regression. {} iterations total.".format(it))

    visualize_f_set(f_set, hist)
    visualize_line(vector[0], vector[1], points, learning_rate.__name__)


def visualize_f_set(f_set, hist):
    def f(w):
        return sum([f_part(w) for f_part in f_set])

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


# Polynomial Regression


class Mode(Enum):
    Simple = 0
    L1 = 1
    L2 = 2
    Elastic = 3


def gamma1():
    return 0.1


def gamma2():
    return 0.1


def simple_f(w):
    return 0.0


def simple_grad_f(w):
    return 0.0


def l1_f(w):
    return gamma1() * sum([abs(wi) for wi in w])


def l1_grad_f(w):
    return gamma1() * np.array([math.copysign(1, wi) for wi in w])


def l2_f(w):
    return gamma2() * sum([wi ** 2 for wi in w])


def l2_grad_f(w):
    return gamma2() * 2.0 * w


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
        upgrade_type=UpgradeType.Empty,
        mode=Mode.Simple
):
    upgrade_core = UpgradeCore(upgrade_type, n + 1)

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
    print(f"{n + 1}: {sum(_f(vector) for _f in f_set)}")
    visualize_curve(vector, points, mode.name)


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


# Learning rate

def const_lr(step):
    return 0.01


def step_lr(step):
    return 0.001 / (step + 1)


def exp_lr(step):
    return 0.1 * math.exp(-0.01 * step - 2)


# Testing functions

def test_f(x):
    return x[0] ** 2 + x[1] ** 2


def test_grad_f(x):
    return np.array([2 * x[0], 2 * x[1]])


# Testing

test_linear_regression(100, 1, exp_lr)