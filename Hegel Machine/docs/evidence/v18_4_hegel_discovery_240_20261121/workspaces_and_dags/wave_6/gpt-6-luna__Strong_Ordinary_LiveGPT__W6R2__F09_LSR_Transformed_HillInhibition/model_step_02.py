import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(0.0, float(row['u_now'])) / 0.1
    s = float(row['s_grad'])
    hill = (u * u) / (1.0 + u * u)  # placeholder; fitted k_half enters below
    return u, s

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    u = np.asarray([max(0.0, float(r['u_now'])) / 0.1 for r in rows])
    s = np.asarray([float(r['s_grad']) for r in rows])
    def residual(p):
        vmax, k_half, alpha, c_leak = p
        pred = vmax * (u*u / (k_half + u*u)) / (1.0 + alpha*s) - c_leak*u
        return pred - y
    best = None
    for start in ([4.0, 1.0, 0.2, 0.1], [5.0, 4.0, 0.5, 0.3], [3.0, 0.2, 1.0, 0.05]):
        result = least_squares(residual, np.asarray(start), bounds=(np.zeros(4), np.full(4, np.inf)), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    vmax, k_half, alpha, c_leak = _PARAMS
    u = max(0.0, float(row['u_now'])) / 0.1
    s = float(row['s_grad'])
    return float(vmax * (u*u / (k_half + u*u)) / (1.0 + alpha*s) - c_leak*u)
