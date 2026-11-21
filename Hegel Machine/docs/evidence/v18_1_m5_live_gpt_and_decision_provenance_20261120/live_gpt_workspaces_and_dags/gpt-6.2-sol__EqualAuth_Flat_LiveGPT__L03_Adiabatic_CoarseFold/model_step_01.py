import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global _k, _a
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        k, a = p
        return x * np.exp(-a * h) - k * u * np.expm1(-a * h) - y

    k0 = np.dot(u, y) / np.dot(u, u)
    result = least_squares(residual, [k0, 3.0 / np.median(h)], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _k, _a = map(float, result.x)


def predict(row):
    z = -_a * float(row['H'])
    return float(row['x_now'] * math.exp(z) - _k * row['u_now'] * math.expm1(z))
