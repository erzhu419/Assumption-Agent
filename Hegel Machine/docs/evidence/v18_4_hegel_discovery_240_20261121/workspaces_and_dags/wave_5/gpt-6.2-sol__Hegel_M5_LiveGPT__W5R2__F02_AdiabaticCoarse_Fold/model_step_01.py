import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global _gain, _rate
    u, x, h, y = np.array([(r['u_now'], r['x_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    gain0 = float(np.dot(u, y) / np.dot(u, u))

    def residual(p):
        gain, rate = p
        t = -rate * h
        return x * np.exp(t) - gain * u * np.expm1(t) - y

    _gain, _rate = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x


def predict(row):
    t = -_rate * row['H']
    return float(row['x_now'] * np.exp(t) - _gain * row['u_now'] * np.expm1(t))