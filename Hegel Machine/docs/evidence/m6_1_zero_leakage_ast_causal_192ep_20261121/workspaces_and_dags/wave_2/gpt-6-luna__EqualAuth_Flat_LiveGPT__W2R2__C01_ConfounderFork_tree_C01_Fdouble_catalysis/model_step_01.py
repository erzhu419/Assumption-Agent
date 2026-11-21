import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, k_half, c_ctrl = p
    x = float(row['x_ind'])
    u = float(row.get('u_ctrl', 0.0))
    q = (x / 1.5) ** 2
    return vmax * q / (k_half + q) - c_ctrl * u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.1]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    params = tuple(float(v) for v in result.x)
    def predict(row):
        return float(_value(row, params))
    return predict

def predict(row):
    return _value(row, (8.0, 1.0, 0.1))