import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * g / (1.0 + g), -u


def fit(train_rows):
    global _coef
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    # Coefficients are vmax, vmax * rho, and leak.
    _coef = least_squares(lambda p: X @ p - y, [6.5, 4.55, 0.5], jac=lambda p: X).x
    return _coef


def predict(row):
    return float(np.dot(_features(row), _coef))
