import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    f = u / (1.2 + u)
    return f, -f * s / (1.0 + s), -math.sqrt(u)


def fit(train_rows):
    global _params
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, [6.0, 4.0, 0.3], jac=lambda p: X, bounds=(0.0, np.inf)).x


def predict(row):
    return float(np.dot(_features(row), _params))
