import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def pred_params(p, r):
        vmax, rho, c_loc = p
        u = float(r['u_now']) / 0.005
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))
    def residual(p):
        return np.array([pred_params(p, r) for r in rows]) - y
    res = least_squares(residual, x0=np.array([4.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 10.0]))
    return res.x

def predict(row):
    vmax, rho, c_loc = fit(_fit_rows)
    u = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))