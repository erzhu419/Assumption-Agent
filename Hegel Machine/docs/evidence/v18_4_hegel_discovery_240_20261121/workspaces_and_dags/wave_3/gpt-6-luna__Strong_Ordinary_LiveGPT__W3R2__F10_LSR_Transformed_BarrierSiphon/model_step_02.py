import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 3.0

def _features(row):
    u = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    barrier = u * math.exp(-_PARAMS[1] / (0.5 + u))
    sink = 1.0 - _PARAMS[2] * s / (1.0 + s)
    return barrier * sink, math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def residual(p):
        j0, beta, rho, c_sqrt = p
        out = []
        for row in rows:
            u = float(row['u_now']) / TAU_SCALE
            s = float(row['s_grad'])
            pred = j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))
            out.append(pred - float(row['y']))
        return np.asarray(out)
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, 1.0, np.inf]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    j0, beta, rho, c_sqrt = _PARAMS
    u = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0)))