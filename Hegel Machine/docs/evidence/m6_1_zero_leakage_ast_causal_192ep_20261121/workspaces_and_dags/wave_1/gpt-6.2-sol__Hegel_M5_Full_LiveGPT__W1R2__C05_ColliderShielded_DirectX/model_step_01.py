import math
import numpy as np
from scipy.optimize import least_squares

_vmax = None
_c_ctrl = None

def fit(train_rows):
    global _vmax, _c_ctrl
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    design = np.column_stack((x / (1.1 + x), u))
    result = least_squares(lambda p: design @ p - y, [1.0, 0.0], jac=lambda p: design)
    _vmax, _c_ctrl = map(float, result.x)

def predict(row) -> float:
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return math.fsum((_vmax * x / (1.1 + x), _c_ctrl * u))
