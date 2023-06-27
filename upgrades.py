from enum import Enum

import numpy as np


# Upgrades

class UpgradeType(Enum):
    Empty = 0,
    Momentum = 1,
    Nesterov = 2,
    AdaGrad = 3,
    RMSProp = 4,
    Adam = 5


def nothing(g_i, it):
    return g_i


class UpgradeCore:
    def __init__(self, upgrade_type: UpgradeType, dim):
        self.eps = 0.000001

        self.upgrade_type = upgrade_type

        self.v = np.array([[0.0] * dim] * 200000)
        self.G = [np.array([0.0000001] * dim)]

        grad_map = {
            UpgradeType.Empty: nothing,
            UpgradeType.Nesterov: self.momentum,
            UpgradeType.Momentum: self.momentum,
            UpgradeType.AdaGrad: self.adagrad,
            UpgradeType.RMSProp: self.rms_prop,
            UpgradeType.Adam: self.adam
        }

        self.main_method = grad_map.get(self.upgrade_type)

    def calculate_grad(self, grad_f, vector, it, lr):
        if self.upgrade_type == UpgradeType.Nesterov:
            grad_f = self.nesterov_grad_f(grad_f, it, lr)

        grad = grad_f(vector)
        return self.main_method(grad, it)

    def nesterov_grad_f(self, grad, it, lr):
        gamma = 0.8
        return lambda w: grad(w - lr * gamma * self.v[it])

    def momentum(self, g_i, it):
        gamma = 0.8
        self.v[it + 1] = gamma * self.v[it] + (1 - gamma) * g_i
        return self.v[it + 1]

    def adagrad(self, g_i, it):
        self.G.append(self.G[it] + np.square(g_i))
        return g_i / np.power(self.G[it + 1], 1 / 2)

    def rms_prop(self, g_i, it):
        gamma = 0.9
        self.G.append(gamma * self.G[it] + (1 - gamma) * np.square(g_i))
        return g_i / np.power(self.G[it + 1], 1 / 2)

    def adam(self, g_i, it):
        gamma1 = 0.9
        gamma2 = 0.999
        self.v[it + 1] = gamma1 * self.v[it] + (1 - gamma1) * g_i
        correct_v = self.v[it + 1] / (1 - gamma1 ** (it + 1))
        self.G.append(gamma2 * self.G[it] + (1 - gamma2) * np.square(g_i))
        correct_G = self.G[it + 1] / (1 - gamma2 ** (it + 1))
        return correct_v / np.power(correct_G, 1 / 2)

# def moving_average(g, gamma):
#     v = np.array([0.0] * (len(g) + 1))
#     for i in range(len(g) - 1):
#         v[i + 1] = gamma * v[i] + (1 - gamma) * g[i]
#     for i in range(len(v) - 1):
#         v[i + 1] = v[i + 1] / (1 - gamma ** (i + 1))
#     v = v[1:]
#     return v
