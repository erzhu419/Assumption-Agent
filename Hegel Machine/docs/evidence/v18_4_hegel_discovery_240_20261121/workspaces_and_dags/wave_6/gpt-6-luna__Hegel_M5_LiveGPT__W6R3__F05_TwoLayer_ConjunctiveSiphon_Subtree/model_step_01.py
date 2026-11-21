import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    u = np.array([r['u_now'] / 0.005 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(np.maximum(u, 0.0))
    result = least_squares(lambda p: pred(p) - y, x0=np.array([6.5, 0.70, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    p = fit([row])
    u = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))