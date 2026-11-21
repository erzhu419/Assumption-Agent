import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 0.3]))
    params = result.x
    def predict(row):
        xv = float(row['x_ind'])
        uv = float(row['u_ctrl'])
        return float(params[0] * xv / (1.1 + xv) + params[1] * uv)
    return predict

def predict(row):
    return fit([])(row)
