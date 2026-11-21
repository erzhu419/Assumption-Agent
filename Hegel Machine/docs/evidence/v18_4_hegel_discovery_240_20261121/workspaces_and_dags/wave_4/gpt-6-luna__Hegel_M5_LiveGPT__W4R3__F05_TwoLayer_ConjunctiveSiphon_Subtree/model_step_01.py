import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.15]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row) -> float:
    p = fit(train_rows)
    return float(_value(row, p))