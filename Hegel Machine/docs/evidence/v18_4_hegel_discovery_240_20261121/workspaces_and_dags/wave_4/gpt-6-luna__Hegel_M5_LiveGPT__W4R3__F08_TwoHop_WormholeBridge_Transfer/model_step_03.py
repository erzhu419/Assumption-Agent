import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, rho, c_loc = p
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_pred(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    return result.x

def predict(row):
    return float(_pred(fit(_train_rows), row))