import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def _value(row, p, sign):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + sign * c_loc * math.sqrt(u)

def fit(train_rows):
    global _MODEL
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for sign in (1.0, -1.0):
        def residual(p):
            return np.array([_value(r, p, sign) for r in train_rows]) - y
        result = least_squares(residual, x0=np.array([5.0, 0.7, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 100.0]), max_nfev=20000)
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, sign, result.x.copy())
    _MODEL = {'sign': best[1], 'params': best[2]}
    return _MODEL

def predict(row) -> float:
    if _MODEL is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _MODEL['params'], _MODEL['sign']))