import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from random import uniform


def generate_n_dots(n, dimensions=2):
    random.seed(12)
    result = []
    for _ in range(n):
        i = []
        for _ in range(dimensions):
            i.append(uniform(-1, 1))
        result.append(i)
    return result


def points_over_function(points, f):
    t = np.linspace(-5, 5, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    ax1.plot(f(points))
    ax2.plot(t, f(t))
    ax2.plot(points, f(points), 'o-')
    plt.show()


def three_dim_plot(f, sz=2):
    t = np.linspace(-sz, sz, 100)
    X, Y = np.meshgrid(t, t)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, f(np.stack((X, Y))))
    plt.show()


def points_over_contour(points, f, levels=30):
    a = max(-np.min(points), np.max(points)) + 0.1
    t = np.linspace(-a, a, 100)
    X, Y = np.meshgrid(t, t)
    fig, ax = plt.subplots()
    ax.contour(X, Y, f(np.stack((X, Y))), levels=levels)
    l, = ax.plot(points[:, 0], points[:, 1], 'o-', markersize=5, alpha=0.8)
    ax.plot(points[-1, 0], points[-1, 1], 'x', markersize=10)
    return ax, l


def multiple_points_over_contour(f, points1, points2=None, points3=None, points4=None, points5=None,
                                 name1="gradient descent", name2="dichotomy",
                                 name3="wolfe", name4=None, name5=None):
    (ax, l1) = points_over_contour(points1, f)
    ls = [l1]
    names = [name1]
    if not (points2 is None):
        l2, = ax.plot(points2[:, 0], points2[:, 1], 'o-', markersize=5, color="r", alpha=0.8)
        ax.plot(points2[-1, 0], points2[-1, 1], 'x', color="r", markersize=10)
        ls.append(l2)
        names.append(name2)
    if not (points3 is None):
        l3, = ax.plot(points3[:, 0], points3[:, 1], 'o-', markersize=5, color="yellowgreen", alpha=0.7)
        ax.plot(points3[-1, 0], points3[-1, 1], 'x', color="yellowgreen", markersize=10)
        ls.append(l3)
        names.append(name3)
    if not (points4 is None):
        l4, = ax.plot(points4[:, 0], points4[:, 1], 'o-', markersize=5, color="violet", alpha=0.7)
        ax.plot(points4[-1, 0], points4[-1, 1], 'x', color="violet", markersize=10)
        ls.append(l4)
        names.append(name4)
    if not (points5 is None):
        l5, = ax.plot(points5[:, 0], points5[:, 1], 'o-', markersize=5, color="orange", alpha=0.7)
        ax.plot(points5[-1, 0], points5[-1, 1], 'x', color="orange", markersize=10)
        ls.append(l5)
        names.append(name5)
    ax.legend(ls, names, loc='upper right', shadow=True)
    plt.title("start: {}".format(points1[0]))
    return plt


def plot_by_two_coordinates(x, y, name, limit=False, graph_name="gradient descent calculations"):
    plt.plot(x, y)
    plt.xlabel(name)
    plt.ylabel(graph_name)
    plt.title(name)
    ax = plt.gca()
    if limit:
        ax.set_ylim([0, 600])
    return plt


def plot_by_array(p_array, start, name_x, name_y, name1="gradient descent", name2="dichotomy", name3="wolfe"):
    l1, = plt.plot(p_array[0][:, 0], p_array[0][:, 1])
    l2, = plt.plot(p_array[1][:, 0], p_array[1][:, 1], color="r", alpha=0.8)
    l3, = plt.plot(p_array[2][:, 0], p_array[2][:, 1], color="yellowgreen", alpha=0.7)
    plt.legend((l1, l2, l3), (name1, name2, name3), loc='upper right', shadow=True)
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title("start: {}".format(start))
    return plt
