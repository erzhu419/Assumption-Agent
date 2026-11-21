import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([6.0, 1.0, 0.2], dtype=float)

def _value(row, p):
    vmax, k_half, c_ctrl = p
    x = float(row['x_ind']) / 1.5
    u = float(row.get('u_ctrl', 0.0))
    q = x * x
    return vmax * q / (k_half + q) - c_ctrl * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    if rows:
        x0 = np.array([6.0, 1.0, 0.2])
        def residual(p):
            return np.array([_value(r, p) - float(r['y']) for r in rows])
        result = least_squares(residual, x0, bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 20.0]))
        _params = result.x
    return predict

def predict(row):
    return float(_value(row, _params))