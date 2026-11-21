import math
import numpy as np
from scipy.optimize import least_squares

def _base(vmax, rho, c, s, u):
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(max(u, 0.0))

def fit(train_rows):
    s = np.array([float(r['s_grad']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, rho, c, k, scrit = p
        return np.array([_base(vmax, rho, c, si, ui) - k * max(0.0, si - scrit) * ui for si, ui in zip(s, u)])
    def resid(p):
        return pred(p) - y
    starts = [
        [6.5, 0.70, 0.1, 2.0, 2.55],
        [5.0, 0.68, 0.6, 1.0, 2.65],
        [8.0, 0.74, -0.3, 3.0, 2.50],
    ]
    best = None
    for p0 in starts:
        res = least_squares(resid, p0, bounds=([0.01, 0.0, -3.0, 0.0, 2.0], [30.0, 0.999, 3.0, 20.0, 3.2]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    p = fit([row])
    vmax, rho, c, k, scrit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_base(vmax, rho, c, s, u) - k * max(0.0, s - scrit) * u)
