import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global _coef
    rows = list(train_rows)
    A = np.array([_features(row) for row in rows])
    y = np.array([float(row['y']) for row in rows])
    _coef = least_squares(lambda p: A @ p - y, [1.0, 0.5, 0.0]).x


def predict(row) -> float:
    return float(np.dot(_features(row), _coef))
