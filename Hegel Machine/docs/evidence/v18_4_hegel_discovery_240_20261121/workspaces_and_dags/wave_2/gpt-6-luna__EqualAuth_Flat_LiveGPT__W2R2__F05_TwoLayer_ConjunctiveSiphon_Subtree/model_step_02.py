import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-8, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
    result = least_squares(lambda p: pred(p) - y, x0=np.array([4.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 10.0]))
    return result.x

def predict(row):
    p = fit(_FIT_ROWS)
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))

_FIT_ROWS = []

def fit(train_rows):
    global _FIT_ROWS
    _FIT_ROWS = list(train_rows)
    u = np.array([max(1e-8, float(r['u_now'])) for r in _FIT_ROWS])
    s = np.array([max(0.0, float(r['s_grad'])) for r in _FIT_ROWS])
    y = np.array([float(r['y']) for r in _FIT_ROWS])
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
    result = least_squares(lambda p: pred(p) - y, x0=np.array([4.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 10.0]))
    return result.x
