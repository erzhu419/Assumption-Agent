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
    design = np.asarray([_features(row) for row in rows], dtype=float)
    target = np.asarray([row['y'] for row in rows], dtype=float)
    _params = least_squares(lambda p: design @ p - target, [8.0, 0.4]).x
    return predict

def predict(row) -> float:
    saturation, control = _features(row)
    vmax, c_ctrl = _params
    return math.fsum((vmax * saturation, c_ctrl * control))