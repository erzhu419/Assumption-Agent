import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x, u, y = np.asarray([(r['x_ind'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float).T
    design = np.column_stack((x / (1.1 + x), u))
    _params = least_squares(lambda p: design @ p - y, [8.0, 0.25], jac=lambda p: design).x
    return _params

def predict(row) -> float:
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return math.fsum((vmax * x / (1.1 + x), c_ctrl * u))
