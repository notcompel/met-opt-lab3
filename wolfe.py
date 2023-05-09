import numpy as np


def generate_wolfe(start_alpha, c1, c2, max_iter):
    def wolfe(vector, p, f: callable, grad_f: callable) -> tuple[float, int, int]:
        alpha = start_alpha

        f_val = f(vector)
        grad_val = grad_f(vector)

        count_f = 1
        count_grad = 1

        for i in range(max_iter):
            count_f += 1

            v_new = vector + alpha * p
            f_val_new = f(v_new)

            if f_val_new <= f_val + c1 * alpha * np.dot(grad_val, p):  # Armijo condition
                count_grad += 1
                if np.dot(grad_f(v_new), p) >= c2 * np.dot(grad_val, p):  # Curvature condition
                    break

            alpha /= 2

        return alpha, count_f, count_grad

    return wolfe
