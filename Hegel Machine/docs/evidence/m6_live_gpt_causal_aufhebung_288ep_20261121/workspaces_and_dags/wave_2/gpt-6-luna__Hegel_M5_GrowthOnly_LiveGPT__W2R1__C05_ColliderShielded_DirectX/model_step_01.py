import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    h = x / (1.1 + x)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * h + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0]))
    return tuple(float(v) for v in result.x)

def predict(row):
    vmax, c_ctrl = fit([row]) if '_fitted_params' not in row else row['_fitted_params']
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
