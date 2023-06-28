import numpy as np
import random


def generate_func(n: int, k: float):
    m = np.random.rand(n, n)
    q, r = np.linalg.qr(m)

    sing_numbers = [random.uniform(1, k) for _ in range(n - 2)]
    sing_numbers.append(k)
    sing_numbers.append(1.0)
    d = np.diag(sing_numbers)

    matrix = np.matmul(np.matmul(q, d), np.transpose(q))
    return matrix
