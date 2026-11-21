import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, row):
        vmax, rho, c_loc = p
        u = max(1e-9, float(row['u_now']))
        s = float(row['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    best = None
    for p0 in ([8.0, 0.7, 0.4], [6.0, 0.5, 1.0], [10.0, 0.8, 0.0]):
        res = least_squares(residual, p0, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(_train_rows)
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))