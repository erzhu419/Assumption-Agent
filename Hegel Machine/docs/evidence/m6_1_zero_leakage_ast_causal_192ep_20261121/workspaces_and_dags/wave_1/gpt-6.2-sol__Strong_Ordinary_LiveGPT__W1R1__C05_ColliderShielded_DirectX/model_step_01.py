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
    design = np.array([_features(row) for row in rows], dtype=float)
    target = np.array([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: design @ p - target, [1.0, 1.0], jac=lambda p: design).x

def predict(row) -> float:
    return math.fsum(float(coef) * value for coef, value in zip(_params, _features(row)))