import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    x0 = np.array([8.0, 0.5, 0.0])
    res = least_squares(lambda p: np.array([_pred(r, p) for r in train_rows]) - y,
                        x0, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    p = res.x
    return lambda row: float(_pred(row, p))

def predict(row):
    return float(_pred(row, _fit_params))