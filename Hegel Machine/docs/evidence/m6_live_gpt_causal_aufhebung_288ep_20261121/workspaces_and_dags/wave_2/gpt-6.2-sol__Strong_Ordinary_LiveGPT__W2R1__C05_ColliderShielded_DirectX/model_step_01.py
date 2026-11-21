import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])


def fit(train_rows):
    global _coef
    rows = list(train_rows)
    design = np.asarray([_features(row) for row in rows], dtype=float)
    target = np.asarray([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda p: design @ p - target, np.zeros(2), jac=lambda p: design).x


def predict(row):
    return float(math.fsum(c * f for c, f in zip(_coef, _features(row))))
