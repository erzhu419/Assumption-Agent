import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global _coefficients
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    # Coefficients are vmax, vmax * rho, and c_loc.
    _coefficients = least_squares(lambda p: X @ p - y, [1.0, 0.5, 0.0], jac=lambda p: X).x


def predict(row) -> float:
    return float(np.dot(_features(row), _coefficients))
