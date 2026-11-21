import math
import numpy as np
from scipy.optimize import least_squares

def _f(row, p):
    vmax, rho, branch = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + branch * math.sqrt(max(u, 0.0))

def fit(train_rows):
    X = np.array([[float(r['u_now']), float(r['s_grad'])] for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_f(r, p) for r in train_rows]) - y
    best = None
    for start in ([5, 0.5, 0.0], [8, 0.7, 0.5], [5, 0.2, -0.5]):
        res = least_squares(residual, start, bounds=([0, 0, -10], [100, 1.5, 10]))
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    p = fit(train_rows)
    return float(_f(row, p))