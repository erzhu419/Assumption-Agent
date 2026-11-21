import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.1
_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    return vmax * (u_n ** 2 / (k_half + u_n ** 2)) / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = ([4.0, 1.0, 0.2, 0.1], [5.0, 4.0, 0.5, 0.05], [3.0, 0.5, 0.05, 0.2])
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=np.asarray(x0), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return {'vmax': float(_params[0]), 'k_half': float(_params[1]), 'alpha_inh': float(_params[2]), 'c_leak': float(_params[3])}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))