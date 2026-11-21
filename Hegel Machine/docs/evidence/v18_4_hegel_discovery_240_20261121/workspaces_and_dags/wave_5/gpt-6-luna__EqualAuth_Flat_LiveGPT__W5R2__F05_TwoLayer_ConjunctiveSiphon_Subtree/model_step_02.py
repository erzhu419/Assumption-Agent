import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    def pred(p, r):
        vmax, rho, c_loc = p
        u = max(1e-8, float(r['u_now']))
        s = max(0.0, float(r['s_grad']))
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([5.0, 0.5, 1.0]), bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]))
    return result.x

def predict(row):
    p = fit(train_rows)
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))