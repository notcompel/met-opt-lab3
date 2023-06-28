import numpy as np


def wolfe_conditions(f, fx, grad, grx, x, d, t, c1=1e-4, c2=0.9):
    grxd = np.dot(grx, d)
    ft = f(x + t * d)
    gxt = np.dot(grad(x + t * d), d)
    armijo = ft <= fx + c1 * t * grxd
    curvature = gxt <= - c2 * grxd
    return armijo and curvature


def wolfe_gradient(f, grad, start, eps=1e-6, c1=1e-4, c2=0.9, alpha=1, max_iter=10000):
    x = np.array(start)

    points = [x]
    func_calc = 0
    grad_calc = 0
    for _ in range(max_iter):
        gr = grad(x)
        t = alpha
        d = -gr
        grad_calc += 2
        func_calc += 2
        fx = f(x)
        while not wolfe_conditions(f, fx, grad, gr, x, d, t, c1, c2):
            t = t / 2
        x = x + t * d
        points.append(x)
        if np.linalg.norm(gr) < eps:
            break
    return np.asarray(points), grad_calc, func_calc
