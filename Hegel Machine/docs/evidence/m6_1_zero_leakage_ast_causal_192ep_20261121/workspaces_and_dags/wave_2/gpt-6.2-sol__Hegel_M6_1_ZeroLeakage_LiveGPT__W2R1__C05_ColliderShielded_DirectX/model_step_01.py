import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([float(row['x_ind']) for row in rows])
    u = np.array([float(row['u_ctrl']) for row in rows])
    y = np.array([float(row['y']) for row in rows])
    design = np.column_stack((x / (1.1 + x), u))
    _params = least_squares(lambda p: design @ p - y, [1.0, 0.0], jac=lambda p: design).x
    return _params


def predict(row) -> float:
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    return math.fsum((vmax * x / (1.1 + x), c_ctrl * float(row['u_ctrl'])))