import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(float(row['u_now']), 1e-8)
    s = max(float(row['s_grad']), 0.0)
    return u, s

def _predict_params(p, row):
    u, s = _features(row)
    offset, amp, alpha, sink, beta = p
    return offset + amp * (u ** alpha) / (1.0 + sink * (s ** beta))

def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(p, r) for r in rows]) - ys
    starts = [
        [0.0, 5.0, 0.5, 0.5, 1.0],
        [0.5, 4.0, 0.3, 1.0, 1.0],
        [0.0, 8.0, 0.7, 0.3, 0.7]
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0,
            bounds=([-10.0, 0.0, 0.01, 0.0, 0.01], [10.0, 100.0, 3.0, 100.0, 3.0]),
            max_nfev=10000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return [float(v) for v in best[1]]

def predict(row):
    return float(_predict_params(_FITTED_PARAMS, row))

_FITTED_PARAMS = [0.0, 5.0, 0.5, 0.5, 1.0]
_orig_fit = fit
def fit(train_rows):
    global _FITTED_PARAMS
    _FITTED_PARAMS = _orig_fit(train_rows)
    return _FITTED_PARAMS
