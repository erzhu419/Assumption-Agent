import math
import numpy as np
from scipy.optimize import least_squares

_tau = 0.1
_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _tau
    hill = (u_n ** 2) / (k_half + u_n ** 2)
    return vmax * hill / (1.0 + alpha_inh * float(row['s_grad'])) - c_leak * u_n

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    starts = [
        [4.0, 1.0, 0.3, 0.03],
        [5.0, 3.0, 0.1, 0.01],
        [3.0, 0.5, 1.0, 0.05],
    ]
    best = None
    for p0 in starts:
        result = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            x0=np.asarray(p0, dtype=float),
            bounds=(np.zeros(4), np.full(4, np.inf)),
            max_nfev=20000,
        )
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1].copy()
    return {'vmax': float(_params[0]), 'k_half': float(_params[1]), 'alpha_inh': float(_params[2]), 'c_leak': float(_params[3])}

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))