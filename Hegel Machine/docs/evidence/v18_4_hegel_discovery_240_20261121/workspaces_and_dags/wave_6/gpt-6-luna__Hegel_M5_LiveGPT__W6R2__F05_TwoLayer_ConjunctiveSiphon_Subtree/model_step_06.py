import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row['u_now']) / 0.05, 1e-8)
    s = float(row['s_grad'])
    return u, s

def _value(p, row):
    j0, beta, rho, c = p
    u, s = _features(row)
    return j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c*math.sqrt(u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    best = None
    for start in ([5, 1, .7, .3], [10, 3, .7, -.2], [3, .1, .5, .8]):
        result = least_squares(residual, np.array(start, dtype=float), bounds=([0, 0, 0, -5], [100, 20, .999, 5]), max_nfev=20000)
        score = float(np.sum(result.fun**2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return {'params': best[1].tolist()}

def predict(row):
    return float(_value(_MODEL['params'], row))
