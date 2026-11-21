import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return x / (1.1 + x), u


def fit(train_rows):
    rows = list(train_rows)
    X = np.asarray([_features(r) for r in rows], dtype=float)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([5.0, 0.7]))
    params = result.x.copy()
    def predict(row):
        a, b = _features(row)
        return float(params[0] * a + params[1] * b)
    return predict


def predict(row):
    # Default parameters from the recalled invariant mechanism; fit() estimates them from training data.
    a, b = _features(row)
    return float(5.0 * a + 0.7 * b)
