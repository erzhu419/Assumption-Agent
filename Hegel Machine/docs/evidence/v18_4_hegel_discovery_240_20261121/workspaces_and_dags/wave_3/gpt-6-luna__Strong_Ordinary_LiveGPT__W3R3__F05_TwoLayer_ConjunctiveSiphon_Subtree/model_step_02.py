import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 4.0
_P = None

def _features(row):
    u = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    base = (1.0 - row_rho_placeholder(s)) * (u / (1.2 + u))
    return u, s, base

def row_rho_placeholder(s):
    # The factor multiplying rho is s/(1+s); the base coefficient is fit below.
    return 0.0

def _model(p, row):
    vmax, rho, c_loc = p
    u = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _P
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    candidates = []
    for start in ([5.0, 0.5, 1.0], [5.0, 0.5, -1.0], [2.0, 0.2, 0.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, -20.0], [100.0, 1.0, 20.0]), max_nfev=20000)
        candidates.append(result)
    best = min(candidates, key=lambda result: float(np.dot(result.fun, result.fun)))
    _P = best.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_P, row))