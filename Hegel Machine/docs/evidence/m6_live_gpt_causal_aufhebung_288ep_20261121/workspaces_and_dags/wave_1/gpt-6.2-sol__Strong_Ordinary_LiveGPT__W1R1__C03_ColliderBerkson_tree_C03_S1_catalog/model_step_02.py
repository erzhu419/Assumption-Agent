import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u2 = float(row['u_2'])
    return math.log1p(float(row['u_1'])), u2 / (1.0 + u2)


def fit(train_rows):
    global _coef
    rows = list(train_rows)
    x = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    _coef = least_squares(lambda c: x @ c - y, np.ones(2), jac=lambda c: x).x


def predict(row):
    return float(np.dot(_features(row), _coef))
