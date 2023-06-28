import numpy as np


def dichotomy_gradient(f, grad, start, eps_g=1e-6, eps_r=0.001, beta=0.2, max_iter=10000):
    x = np.array(start)
    points = [x]

    func_calc = 0
    for _ in range(max_iter):
        a = 1e-4
        b = beta
        gr = grad(x)
        while abs(b - a) > 2 * eps_r:
            c = (a + b) / 2
            func_calc += 2
            if f(x - ((c + eps_r) * gr)) < f(x - ((c - eps_r) * gr)):
                a = c
            else:
                b = c

        x = x - ((a + b) / 2.0) * gr
        points.append(x)
        if np.linalg.norm(gr) < eps_g:
            break
    return np.asarray(points), len(points), func_calc
