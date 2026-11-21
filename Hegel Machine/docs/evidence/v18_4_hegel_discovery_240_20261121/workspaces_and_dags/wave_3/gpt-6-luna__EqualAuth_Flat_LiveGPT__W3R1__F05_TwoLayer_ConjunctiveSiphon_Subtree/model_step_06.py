import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.05

def _pred(row, p, sign):
    vmax, rho, c_loc = p
    u = max(1e-8, float(row['u_now']) / TAU)
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + sign * c_loc * math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    best = None
    for sign in (1.0, -1.0):
        def residual(p):
            return np.array([_pred(r, p, sign) for r in rows]) - y
        res = least_squares(residual, x0=np.array([5.0, 0.7, 0.2]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 20.0]), max_nfev=20000)
        score = float(np.mean(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x, sign)
    return {'params': best[1].tolist(), 'sign': best[2]}

def predict(row):
    p = fit_params['params']
    return float(_pred(row, p, fit_params['sign']))