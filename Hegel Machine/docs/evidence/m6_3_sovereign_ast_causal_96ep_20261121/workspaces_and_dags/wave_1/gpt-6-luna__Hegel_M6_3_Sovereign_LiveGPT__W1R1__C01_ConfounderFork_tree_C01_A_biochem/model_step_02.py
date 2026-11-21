import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    z = np.array([float(r['z_1']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    gate = (z > 2.5).astype(float)
    def residual(p):
        vmax, k, c_ctrl, bonus = p
        xx = (x / 1.5) ** 2
        pred = vmax * xx / (k + xx) - c_ctrl * u + bonus * gate
        return pred - y
    res = least_squares(residual, x0=[8.0, 1.0, 0.4, 2.0], bounds=([0.01, 0.01, -5.0, -10.0], [30.0, 30.0, 5.0, 10.0]))
    _PARAMS = tuple(float(v) for v in res.x)
    return predict

def predict(row):
    vmax, k, c_ctrl, bonus = _PARAMS
    xx = (float(row['x_ind']) / 1.5) ** 2
    gate = 1.0 if float(row['z_1']) > 2.5 else 0.0
    return float(vmax * xx / (k + xx) - c_ctrl * float(row['u_ctrl']) + bonus * gate)
