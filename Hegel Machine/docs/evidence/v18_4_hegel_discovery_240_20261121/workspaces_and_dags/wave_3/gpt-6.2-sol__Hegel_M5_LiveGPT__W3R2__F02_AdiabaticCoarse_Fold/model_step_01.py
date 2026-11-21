import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global _gain, _rate
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(params):
        gain, rate = params
        eq = gain * u
        return eq + (x - eq) * np.exp(-rate * h) - y

    gain0 = np.dot(u, y) / np.dot(u, u)
    _gain, _rate = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]), gtol=1e-12).x


def predict(row):
    eq = _gain * row['u_now']
    return float(eq + (row['x_now'] - eq) * math.exp(-_rate * row['H']))
