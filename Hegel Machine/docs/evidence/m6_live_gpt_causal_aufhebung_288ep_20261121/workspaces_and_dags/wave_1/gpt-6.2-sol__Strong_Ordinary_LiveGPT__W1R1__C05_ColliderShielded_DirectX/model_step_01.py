import numpy as np
from scipy.optimize import least_squares


def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])


def fit(train_rows):
    global _coef
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([float(row['y']) for row in rows], dtype=float)
    _coef = least_squares(lambda p: X @ p - y, [1.0, 1.0], jac=lambda p: X).x


def predict(row) -> float:
    valve, ctrl = _features(row)
    return float(_coef[0] * valve + _coef[1] * ctrl)
