import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0]))
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    params = fit([row]) if False else None
    raise RuntimeError('Call fit(train_rows) first; use the returned parameters in a fitted predictor.')