import matplotlib.pyplot as plt
import numpy as np
from lab1.plot import plotter
import methods

np.random.seed(123)


def f(x):
    return 4 * x[0] ** 2 + 9 * x[1] ** 2 - 4 * x[0] * x[1] - 2 * x[0] + 12 * x[1] + 7


def grad(x):
    h = 1e-5
    N = 2
    return (f(x[:, np.newaxis] + h * np.eye(N)) - f(x[:, np.newaxis] - h * np.eye(N))) / (2 * h)


def print_info(name, start, points, grad_calc, func_calc):
    print("start:", start)
    print("{}:".format(name))
    print(points[-1], f(points[-1]))
    print("gradient calculations:", grad_calc)
    print("function calculations:", func_calc)


def plot_from_start(start):
    (points1, grad_calc1, func_calc1) = methods.bfgs(f, grad, start)
    print_info("BFGS", start, points1, grad_calc1, func_calc1)
    (points2, grad_calc2, func_calc2) = methods.l_bfgs(f, grad, start)
    print_info("L_BFGS", start, points2, grad_calc2, func_calc2)

    plotter.multiple_points_over_contour(f, points1, name1="BFGS").show()
    plotter.multiple_points_over_contour(f, points2, name1="L_BFGS").show()


plot_from_start([10, -5])