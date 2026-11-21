import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU = 0.1

def _features(row):
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return u, s

def _law(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u, s = _features(row)
    hill = (u * u) / (k_half + u * u)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(logp):
        p = np.exp(logp)
        return np.asarray([_law(r, p) for r in train_rows]) - ys
    starts = [np.log([4.0, 1.0, 0.5, 0.05]), np.log([8.0, 4.0, 0.1, 0.01]), np.log([3.0, 0.2, 2.0, 0.1])]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=(np.log([1e-8]*4), np.log([1e4]*4)), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = np.exp(best[1])
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_law(row, _params))