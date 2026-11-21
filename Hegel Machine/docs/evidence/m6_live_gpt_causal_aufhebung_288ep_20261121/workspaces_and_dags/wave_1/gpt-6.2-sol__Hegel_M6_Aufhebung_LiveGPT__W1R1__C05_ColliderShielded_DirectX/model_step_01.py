import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])


def fit(train_rows):
    global _coef
    rows = list(train_rows)
    features = np.array([_features(row) for row in rows], dtype=float)
    targets = np.array([float(row['y']) for row in rows])
    _coef = least_squares(lambda p: features @ p - targets, np.ones(2), jac=lambda p: features).x


def predict(row) -> float:
    return math.fsum(a * b for a, b in zip(_features(row), _coef))
