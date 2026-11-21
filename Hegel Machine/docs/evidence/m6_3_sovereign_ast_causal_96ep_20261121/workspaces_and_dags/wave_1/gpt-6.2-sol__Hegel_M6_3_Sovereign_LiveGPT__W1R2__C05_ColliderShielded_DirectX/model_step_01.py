import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    design = np.column_stack((x / (1.1 + x), u))
    _params = least_squares(lambda p: design @ p - y, [5.85, 0.30], jac=lambda p: design).x
    return predict


def predict(row) -> float:
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return math.fsum((_params[0] * x / (1.1 + x), _params[1] * u))
