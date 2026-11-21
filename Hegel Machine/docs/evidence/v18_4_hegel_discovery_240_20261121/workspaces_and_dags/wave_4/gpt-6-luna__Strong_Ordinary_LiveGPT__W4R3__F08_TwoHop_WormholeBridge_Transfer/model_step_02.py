import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p, r):
        vmax, rho, c_loc = p
        u = max(0.0, float(r['u_now']))
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.5, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 5.0, 10.0]))
    return result.x

def predict(row):
    vmax, rho, c_loc = fit(_train_rows)
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))