import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    x = float(row['x_ind'])
    return x / (1.1 + x), float(row['u_ctrl'])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, [1.0, 0.0], jac=lambda p: X).x

def predict(row) -> float:
    return math.fsum(p * f for p, f in zip(_params, _features(row)))
