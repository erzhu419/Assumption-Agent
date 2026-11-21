import numpy as np
from scipy.optimize import least_squares


def _features(row):
    return ((row['u_now'] - row['u_past']) / 0.02, row['u_now'] / 0.02, row['H'])


def fit(train_rows):
    global _params
    step, u, h = np.asarray([_features(row) for row in train_rows], dtype=float).T
    y = np.asarray([row['y'] for row in train_rows], dtype=float)

    def residual(p):
        gain, rate, equilibrium = p
        return gain * step * np.exp(-rate * h) + equilibrium * u - y

    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params


def predict(row) -> float:
    step, u, h = _features(row)
    gain, rate, equilibrium = _params
    return float(gain * step * np.exp(-rate * h) + equilibrium * u)
