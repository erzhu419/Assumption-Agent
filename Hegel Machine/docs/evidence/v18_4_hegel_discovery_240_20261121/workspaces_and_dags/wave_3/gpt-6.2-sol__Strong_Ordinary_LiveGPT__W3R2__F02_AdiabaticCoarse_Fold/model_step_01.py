import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global _gain, _rate
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        eq = p[0] * u
        return eq + (x - eq) * np.exp(-p[1] * h) - y

    result = least_squares(residual, [np.dot(u, y) / np.dot(u, u), 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]), gtol=1e-13)
    _gain, _rate = map(float, result.x)


def predict(row) -> float:
    eq = _gain * row['u_now']
    return float(eq + (row['x_now'] - eq) * math.exp(-_rate * row['H']))
