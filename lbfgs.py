from copy import copy

import numpy as np
import scipy as sp

from visualization import visualize
from wolfe import generate_wolfe, wolfe_gradient


class LBFGS:
    def __init__(self, f, grad_f, start_pos, learning_rate):
        self.m = 100

        self.f = f
        self.grad_f = grad_f
        self.start_pos = start_pos
        self.learning_rate = learning_rate
        self.wolfe = generate_wolfe(5.0, 0.0001, 0.9, 1000)

    def process(self):
        hist = self.calculate_history()
        visualize(self.f, hist)

    def calculate_history(self) -> np.array:
        vector = copy(self.start_pos)
        hist = [tuple(vector)]

        eps = 0.0001
        it = 0

        I = np.eye(len(vector))

        S = []
        Y = []

        while it < 10000 and end_condition(hist, eps, vector):

            grad_v = self.grad_f(vector)

            if it > self.m:
                gamma = np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1])
                Hk0 = gamma * I
            else:
                Hk0 = I

            q = grad_v
            i = len(S) - 1
            while i >= 0:
                yi = Y[i]
                si = S[i]
                ro = 1.0 / (np.dot(yi, si))
                alpha = ro * np.dot(si, q)
                q -= alpha * yi
                i -= 1

            r = np.matmul(q, Hk0)
            i = 0
            while i != len(S):
                yi = Y[i]
                si = S[i]
                ro = 1.0 / (np.dot(yi, si))
                alpha = ro * np.dot(si, q)
                beta = ro * np.dot(yi, r)
                r += si * (alpha - beta)
                i += 1

            p = -r
            # lr = wolfe_gradient(self.f, self.grad_f, vector)[0]
            # next_vector = wolfe_gradient(self.f, self.grad_f, vector)[0]
            lr = sp.optimize.line_search(self.f, self.grad_f, vector, p)[0]
            if lr is None:
                break

            next_vector = vector + lr * p
            next_grad = self.grad_f(next_vector)

            s = next_vector - vector
            y = next_grad - grad_v

            if len(S) == self.m:
                S.pop(0)
                Y.pop(0)
            S.append(s)
            Y.append(y)

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
