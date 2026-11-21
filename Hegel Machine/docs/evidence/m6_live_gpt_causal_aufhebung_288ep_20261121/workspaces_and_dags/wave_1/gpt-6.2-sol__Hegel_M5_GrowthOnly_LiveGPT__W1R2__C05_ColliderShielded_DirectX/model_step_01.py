import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _features(row):
    x = float(row['x_ind'])
    return x / math.fsum((1.1, x)), float(row['u_ctrl'])


def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.zeros(2), jac=lambda p: X).x
    return predict


def predict(row):
    opening, control = _features(row)
    return float(_params[0] * opening + _params[1] * control)
