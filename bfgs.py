from copy import copy

import numpy as np

from visualization import visualize
from wolfe import generate_wolfe


class BFGS:
    def __init__(self, f, grad_f, start_pos):
        self.f = f
        self.grad_f = grad_f
        self.start_pos = start_pos
        self.wolfe = generate_wolfe(0.9, 0.0001, 0.9, 1000)

    def process(self):
        hist = self.calculate_history()
        # visualize(self.f, hist)
        return hist[-1]

    def calculate_history(self) -> np.array:
        vector = copy(self.start_pos)
        hist = [tuple(vector)]

        eps = 0.0001
        it = 0

        I = np.eye(len(vector))
        H = I

        while it < 10000 and end_condition(hist, eps, vector):
            grad_v = self.grad_f(vector)
            p = -np.dot(H, grad_v)
            lr = self.wolfe(vector, p, self.f, self.grad_f)[0]

            print(lr)

            next_vector = vector + lr * p
            next_grad = self.grad_f(next_vector)

            s = next_vector - vector
            y = next_grad - grad_v

            ro = 1.0 / (np.dot(y, s))
            A1 = I - ro * np.outer(s, y)
            A2 = I - ro * np.outer(y, s)
            H = np.dot(A1, np.dot(H, A2)) + (ro * s[:, np.newaxis] *
                                             s[np.newaxis, :])

            vector = next_vector
            hist.append(tuple(vector))

            log_steps(it, hist, vector)
            it += 1

        print("Finished. {} iterations total.".format(it))
        print("Minimum is in {}".format([round(x, 4) for x in hist[-1]]))
        return np.array(hist)


def end_condition(hist, eps, vector):
    return len(hist) <= 2 or any(abs(hist[-2][i] - hist[-1][i]) >= eps for i in range(len(vector)))


def log_steps(it, hist, vector):
    print(f"{it}: {' '.join(f'%.8f' % hist[-1][i] for i in range(len(vector)))}")
