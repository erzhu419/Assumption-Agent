import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p, branch):
    vmax, rho, c = p
    un = max(0.0, float(row['u_now']) / 4.0)
    sg = float(row['s_grad'])
    base = vmax * (1.0 - rho * sg / (1.0 + sg)) * (un / (1.2 + un))
    if branch == 'sqrt_plus':
        return base + c * math.sqrt(un)
    if branch == 'sqrt_minus':
        return base - c * math.sqrt(un)
    return base - c * un

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    best = None
    for branch in ('sqrt_plus', 'sqrt_minus', 'leak'):
        def residual(p):
            return np.asarray([_predict_params(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, x0=[5.0, 0.4, 0.1], bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]))
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x, branch)
    _PARAMS = (best[1], best[2])
    return predict

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _PARAMS[0], _PARAMS[1]))
