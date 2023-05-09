from enum import Enum

import numpy as np


# Upgrades

def nothing(g_i, it):
    return g_i


class UpgradeCore:
    def __init__(self, dim):
        self.eps = 0.000001

        self.main_method = nothing

    def calculate_grad(self, grad_f, vector, it, lr):
        grad = grad_f(vector)
        return self.main_method(grad, it)
