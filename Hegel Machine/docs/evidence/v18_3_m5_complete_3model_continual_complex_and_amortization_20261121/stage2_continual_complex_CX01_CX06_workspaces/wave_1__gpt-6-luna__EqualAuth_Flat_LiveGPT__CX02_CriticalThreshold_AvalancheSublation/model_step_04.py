import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]))
    return result.x

def predict(row):
    return float(_value(row, fit(_FIT_ROWS)))