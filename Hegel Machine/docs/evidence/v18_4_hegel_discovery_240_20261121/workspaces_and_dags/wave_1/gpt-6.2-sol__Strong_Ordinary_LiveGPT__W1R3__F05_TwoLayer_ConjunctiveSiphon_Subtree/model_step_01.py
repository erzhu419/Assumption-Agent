import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global _params
    X = np.asarray([_features(row) for row in train_rows], dtype=float)
    y = np.asarray([row['y'] for row in train_rows], dtype=float)
    # Coefficients are vmax, vmax*rho, and the signed square-root amplitude.
    _params = least_squares(lambda p: X @ p - y, np.ones(3), jac=lambda p: X).x


def predict(row) -> float:
    return float(np.dot(_features(row), _params))
