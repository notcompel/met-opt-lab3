import math
import tracemalloc
import numpy as np
from math import sqrt
from DanyaLab1.method import wolfe_gradient
import time


def gauss_newton(f, jac, x, y, p0, eps=1e-4, max_iter=1000):
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[1]
    start_time = time.time()
    jac_calc = 0
    func_calc = 0
    p = p0
    points = [np.asarray(p)]
    for itr in range(max_iter):
        J = jac(p)
        jac_calc += 1
        r = y - f(p)
        func_calc += 1
        pseudo_inverse = np.linalg.pinv(J)
        delta = np.dot(pseudo_inverse, r)
        p += delta
        if np.linalg.norm(delta) < eps:
            break
        points.append(p)
    memory = tracemalloc.get_traced_memory()[1] - memory_start
    tracemalloc.stop()
    return np.asarray(points), jac_calc, func_calc, time.time() - start_time, memory


def dogleg_method(gk, hes, trust_radius):
    pb = -np.dot(np.linalg.inv(hes), gk)
    norm_pb = np.linalg.norm(pb)

    if norm_pb <= trust_radius:
        return pb

    pu = - (np.dot(gk, gk) / np.dot(gk, np.dot(hes, gk))) * gk
    dot_pu = np.dot(pu, pu)
    norm_pu = sqrt(dot_pu)

    if norm_pu >= trust_radius:
        return trust_radius * pu / norm_pu

    pb_pu = pb - pu
    dot_pb_pu = np.dot(pb_pu, pb_pu)
    dot_pu_pb_pu = np.dot(pu, pb_pu)
    fact = dot_pu_pb_pu ** 2 - dot_pb_pu * (dot_pu - trust_radius ** 2)
    tau = (-dot_pu_pb_pu + sqrt(fact)) / dot_pb_pu

    return pu + tau * pb_pu


def trust_region_dogleg(f, grad, hess, start, initial_trust_radius=1.0, max_trust_radius=100.0, eta=0.15, eps=1e-4,
                        max_iter=20):
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[1]
    start_time = time.time()
    xk = start
    points = [xk]
    trust_radius = initial_trust_radius
    func_calc = 0
    grad_calc = 0
    for k in range(max_iter):

        gk = grad(xk)
        grad_calc += 1
        hes = hess(xk)
        grad_calc += 1

        pk = dogleg_method(gk, hes, trust_radius)

        actual_reduction = f(xk) - f(xk + pk)
        func_calc += 2

        prediction_reduction = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(hes, pk)))
        rhok = 1e99
        if prediction_reduction != 0.0:
            rhok = actual_reduction / prediction_reduction

        norm_pk = sqrt(np.dot(pk, pk))

        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        elif rhok > 0.75 and norm_pk == trust_radius:
            trust_radius = min(2.0 * trust_radius, max_trust_radius)

        if rhok > eta:
            xk = xk + pk
        points.append(xk)
        if np.linalg.norm(gk) < eps:
            break
    memory = tracemalloc.get_traced_memory()[1] - memory_start
    tracemalloc.stop()
    return np.asarray(points), grad_calc, func_calc, time.time() - start_time, memory


def bfgs(f, grad, start, eps=1e-4, max_iter=10000):
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[1]
    start_time = time.time()
    x = np.array(start)
    points = [x]
    grad_calc = 1
    func_calc = 0
    gx = grad(x)

    E = np.eye(len(x))
    H = E

    for epoch in range(max_iter):
        pk = -np.dot(H, gx)

        lr = 1
        fx = f(x)

        func_calc += 2
        grad_calc += 1
        while not wolfe_gradient.wolfe_conditions(f, fx, grad, gx, x, pk, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2

        xn = x + lr * pk
        s = xn - x
        x = xn

        gradn = grad(xn)
        grad_calc += 1

        y = gradn - gx
        gx = gradn
        ro = 1.0 / (np.dot(y, s))
        С1 = E - ro * s[:, np.newaxis] * y[np.newaxis, :]
        С2 = E - ro * y[:, np.newaxis] * s[np.newaxis, :]
        H = np.dot(С1, np.dot(H, С2)) + (ro * s[:, np.newaxis] * s[np.newaxis, :])

        points.append(x)
        if np.linalg.norm(gx) < eps:
            break
    memory = tracemalloc.get_traced_memory()[1] - memory_start
    tracemalloc.stop()
    return np.asarray(points), grad_calc, func_calc, time.time() - start_time, memory


def l_bfgs(f, grad, start, eps=1e-4, max_iter=10000, m=10):
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[1]
    start_time = time.time()
    xk = np.array(start)
    I = np.identity(len(xk))
    Hk = I
    grad_calc = 0
    func_calc = 0
    funcs = []
    grads = []
    points = [xk]

    def calculate_pk(H0, p):
        m_t = len(funcs)
        q = p
        a = np.zeros(m_t)
        b = np.zeros(m_t)
        for i in reversed(range(m_t)):
            s = funcs[i]
            y = grads[i]
            rho_i = float(1.0 / y.T.dot(s))
            a[i] = rho_i * s.dot(q)
            q = q - a[i] * y

        r = H0.dot(q)

        for i in range(m_t):
            s = funcs[i]
            y = grads[i]
            rho_i = float(1.0 / y.T.dot(s))
            b[i] = rho_i * y.dot(r)
            r = r + s * (a[i] - b[i])

        return r

    for i in range(max_iter):
        # compute search direction
        gk = grad(xk)
        grad_calc += 1
        pk = -calculate_pk(I, gk)

        fx = f(xk)
        func_calc += 2
        grad_calc += 1
        lr = 1
        while not wolfe_gradient.wolfe_conditions(f, fx, grad, gk, xk, pk, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2

        xk1 = xk + lr * pk
        gk1 = grad(xk1)
        grad_calc += 1

        sk = xk1 - xk
        yk = gk1 - gk

        funcs.append(sk)
        grads.append(yk)
        if len(funcs) > m:
            funcs = funcs[1:]
            grads = grads[1:]
        points.append(xk)

        if np.linalg.norm(xk1 - xk) < eps:
            xk = xk1
            break

        xk = xk1
    memory = tracemalloc.get_traced_memory()[1] - memory_start
    tracemalloc.stop()
    return np.asarray(points), grad_calc, func_calc, time.time() - start_time, memory
