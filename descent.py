from copy import copy

import numpy as np

from visualization import visualize
from upgrades import UpgradeCore


class GradientDescent:
    def __init__(self, f, grad_f, start_pos, learning_rate):
        self.f = f
        self.grad_f = grad_f
        self.start_pos = start_pos
        self.learning_rate = learning_rate

    def process(self):
        core = UpgradeCore(len(self.start_pos))

        hist = self.calculate_history(core)

        visualize(self.f, hist)

    def calculate_history(self, upgrade_core: UpgradeCore) -> np.array:
        vector = copy(self.start_pos)
        hist = [tuple(vector)]

        eps = 0.001
        it = 0

        while it < 10000 and end_condition(hist, eps, vector):
            if it > 10 and jump_condition(hist, eps, vector):
                print("Спуск не сошелся.")
                break

            lr = self.learning_rate(it)
            grad_v = upgrade_core.calculate_grad(self.grad_f, vector, it, lr)
            vector -= lr * grad_v

            hist.append(tuple(vector))
            if it % 10 == 0:
                log_steps(it, hist, vector)
            it += 1

        print("Finished. {} iterations total.".format(it))
        print("Minimum is in {}".format([round(x, 4) for x in hist[-1]]))
        return np.array(hist)


def end_condition(hist, eps, vector):
    return len(hist) <= 2 or any(abs(hist[-2][i] - hist[-1][i]) >= eps for i in range(len(vector)))


def jump_condition(hist, eps, vector):
    return any(all(abs(hist[-k][i] - hist[-1][i]) < eps for i in range(len(vector))) for k in range(3, 10))


def log_steps(it, hist, vector):
    print(f"{it}: {' '.join(f'%.8f' % abs(hist[-1][i]) for i in range(len(vector)))}")