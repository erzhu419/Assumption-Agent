import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])


def fit(train_rows):
    rows = list(train_rows)
    a = np.asarray([_features(r) for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    global _coef
    _coef = least_squares(lambda p: a @ p - y, [0.0, 0.0], jac=lambda p: a).x


def predict(row) -> float:
    return math.fsum(float(c) * f for c, f in zip(_coef, _features(row)))
