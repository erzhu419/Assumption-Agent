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
    design = np.asarray([_features(row) for row in rows], dtype=float)
    y = np.asarray([float(row['y']) for row in rows], dtype=float)
    _params = least_squares(lambda p: design @ p - y, [1.0, 0.0], jac=lambda p: design).x

def predict(row) -> float:
    return float(np.dot(_params, _features(row)))
