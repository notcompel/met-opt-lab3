import timeit
import methods
import numpy as np


def generate_points(n, dim, f):
    X = np.random.rand(n, dim)
    y = []
    X_err = X + 0.2 * np.random.randn(n, dim)
    for x in X_err:
        y.append(f(x))
    return X, np.asarray(y)


def mse(X, y, w):
    res = 0
    for i in range(0, len(y)):
        x = 0
        for j in range(0, len(X[i])):
            x += X[i][j] * w[j]
        res += np.square(y[i] - x)
    return res / len(y)


def mse_func(X, y):
    return lambda w: mse(X, y, w)


def grad_calculator(x, func, n):
    h = 1e-5
    res = []
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = h
        res.append((func(x + delta) - func(x - delta)) / (2 * h))
    return np.asarray(res)


def grad_func(f, n):
    return lambda x: grad_calculator(x, f, n)


n = 1000
dim = 2
(X, y) = generate_points(n, dim, lambda x: 5 * x[0] + 2 * x[1])
f = mse_func(X, y)
grad = grad_func(f, dim)

starts = [[0, 0], [-1, 2], [-10, 0]]

def wrap():
    methods.sgd_adam(f, grad, start, learning_rate=lambda epoch: 0.5)

for i in starts:
    start = i
    execution_time = timeit.timeit(wrap, number=10)
    print(f"Время выполнения {start[0]}, {start[1]} : {execution_time / 10:.6f}s")


