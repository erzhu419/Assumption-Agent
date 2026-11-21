import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 0.1

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u_n = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    hill = (u_n ** 2) / (k_half + u_n ** 2)
    return vmax * hill / (1.0 + alpha_inh * s) - c_leak * u_n

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    starts = [
        [4.0, 1.0, 0.5, 0.01],
        [5.0, 4.0, 0.2, 0.05],
        [3.0, 0.5, 1.0, 0.001],
        [8.0, 10.0, 0.1, 0.1],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0=np.asarray(x0), bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=20000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))