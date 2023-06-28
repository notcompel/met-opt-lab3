import numpy as np


def gradient_descent(f, grad, start, eps=1e-6, lr=0.2, max_iter=10000):
    x = np.array(start)

    points = [x]
    for _ in range(max_iter):
        gr = grad(x)
        x = x - lr * gr
        points.append(x)
        if np.linalg.norm(gr) < eps:
            break
    return np.asarray(points), len(points), 0
