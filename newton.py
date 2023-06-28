import numpy as np
import matplotlib.pyplot as plt


# Solves Ax = b
def conj_grad(A, b):
    eps = 0.00001

    x = np.array([0.0] * len(A))
    r = A.dot(x) - b
    p = -r

    step = 0

    while np.linalg.norm(r) > eps:
        step += 1
        if step % 10000 == 0:
            print(f'[conj_grad] Step {step} ({round(np.linalg.norm(r), 6)})')

        APK = A.dot(p)

        alpha = r.dot(p) / p.dot(APK)
        x -= alpha * p

        r = A.dot(x) - b
        beta = r.dot(APK) / p.dot(APK)
        p = -r + beta * p

    # print(f'[conj_grad] Step {step} ({round(abs(sum(r)), 6)})')
    return x


def gauss_newton(f, J, x0, max_iter=1000, tol=1e-10):
    """
    f: function that maps x to m-dimensional vector of residuals
    J: function that maps x to (m x n) Jacobian matrix of f
    x0: initial guess for parameter vector (n-dimensional)
    b: observed data (m-dimensional)
    max_iter: maximum number of iterations
    tol: tolerance for convergence
    """

    x = x0.copy()
    step = 0
    while step < max_iter:
        Jx = J(x)
        r = f(x)

        # (J^T J) delta = -J^T r
        # Ax = b
        A = np.matmul(Jx.T, Jx)
        b = -np.matmul(Jx.T, r)
        delta = conj_grad(A, b)

        x += delta

        if np.linalg.norm(delta) < tol:
            print("[Gauss-Newton] Converged!")
            break

        step += 1
        print(f"[Gauss-Newton] Step {step} ({np.linalg.norm(delta)})")

    return x


def f1(x, ab):
    a, b = ab
    return a * np.exp(-b * x)


def f2(x, ab):
    a, b = ab
    return a * np.sin(-b * x)


def f3(x, ab):
    a, b = ab
    return a * np.exp(-b * x) + a * np.sin(-b * x)


def private_residual(v, f, x_data, y_data):
    return y_data - f(x_data, v)


def jacobian(v, residual):
    eps = 1e-6
    real = residual(v)

    J = []
    for i in range(len(v)):
        v_shifted = v.copy()
        v_shifted[i] += eps
        shifted = residual(v_shifted)

        J.append((shifted - real) / eps)
    J = np.array(J).T

    return J


def test():
    f = f1

    x_data = np.linspace(0, 1, 20)
    ab_true = np.array([1.5, 5.0])
    y_data = f(x_data, ab_true) + np.random.normal(scale=0.03, size=len(x_data))

    ab_start = np.array([0.5, 1.5])

    def residual(v):
        return private_residual(v, f, x_data, y_data)

    x_opt = gauss_newton(residual, lambda v: jacobian(v, residual), ab_start)

    # Print the optimized parameters
    print(f"Optimized parameters: a = {round(x_opt[0], 5)}, b = {round(x_opt[1], 5)}")

    plt.scatter(x_data, y_data)
    plt.plot(x_data, f(x_data, ab_true), label="True")
    plt.plot(x_data, f(x_data, x_opt), label="Fit")
    plt.legend()
    plt.show()
