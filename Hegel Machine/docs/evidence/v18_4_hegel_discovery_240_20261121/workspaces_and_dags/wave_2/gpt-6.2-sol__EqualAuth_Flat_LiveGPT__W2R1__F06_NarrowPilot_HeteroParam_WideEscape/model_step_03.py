import math
import numpy as np
from scipy.optimize import least_squares


def features(row):
    u, s = float(row['u_now']), float(row['s_grad'])
    sat = u / (1.2 + u)
    return sat, -sat * s / (1.0 + s), math.sqrt(u)


def fit(train_rows):
    global coefficients
    rows = list(train_rows)
    X = np.asarray([features(r) for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    coefficients = least_squares(lambda p: X @ p - y, [1.0, 1.0, 0.0], jac=lambda p: X).x


def predict(row):
    return float(np.dot(features(row), coefficients))
