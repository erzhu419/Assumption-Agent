import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    z3 = np.array([float(r['z_3']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    gate = (z3 > 0.6).astype(float)
    xx = (x / 1.5) ** 2
    def residual(p):
        vmax, k, c_ctrl, indicator = p
        return vmax * xx / (k + xx) - c_ctrl * u + indicator * gate - y
    best = None
    for initial in ([8.0, 1.0, 0.4, 2.0], [10.0, 2.0, 0.0, 0.0], [6.0, 0.5, -1.0, 3.0]):
        result = least_squares(residual, x0=initial,
                               bounds=([0.01, 0.01, -5.0, -20.0], [30.0, 30.0, 5.0, 20.0]),
                               max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = tuple(float(v) for v in best.x)
    return predict

def predict(row):
    vmax, k, c_ctrl, indicator = _PARAMS
    xx = (float(row['x_ind']) / 1.5) ** 2
    gate = 1.0 if float(row['z_3']) > 0.6 else 0.0
    return float(vmax * xx / (k + xx) - c_ctrl * float(row['u_ctrl']) + indicator * gate)
