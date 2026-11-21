import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        vmax, rho, c = p
        u = max(0.0, float(r['u_now']))
        s = max(0.0, float(r['s_grad']))
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(u)
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    best = None
    for p0 in ([3.0, 0.7, 0.0], [4.0, 0.5, 0.2], [2.0, 0.9, -0.2]):
        res = least_squares(residual, p0, bounds=([0.0, 0.0, -5.0], [20.0, 1.0, 5.0]))
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    vmax, rho, c = fit(_TRAIN_ROWS)
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(u))