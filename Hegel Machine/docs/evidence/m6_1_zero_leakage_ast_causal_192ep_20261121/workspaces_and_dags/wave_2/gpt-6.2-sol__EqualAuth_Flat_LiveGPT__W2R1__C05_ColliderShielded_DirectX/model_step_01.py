import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    x, u, y = np.array([(r['x_ind'], r['u_ctrl'], r['y']) for r in rows], dtype=float).T
    design = np.column_stack((x / (1.1 + x), u))
    _params = least_squares(lambda p: design @ p - y, [1.0, 0.0], jac=lambda p: design).x
    return _params

def predict(row) -> float:
    x = float(row['x_ind'])
    vmax, c_ctrl = _params
    return float(vmax * x / (1.1 + x) + c_ctrl * float(row['u_ctrl']))