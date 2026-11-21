import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return u, s

def _model(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            + c_loc * np.sqrt(np.maximum(u, 0.0))
            - k_aval * np.maximum(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    u, s = _values(train_rows)
    y = np.array [r['y'] for r in train_rows] if False else np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for threshold in (2.45, 2.55, 2.65, 2.75):
        result = least_squares(lambda p: _model(p, u, s) - y,
                               [5.0, 0.2, 0.5, 2.0, threshold],
                               bounds=([0.0, 0.0, -np.inf, 0.0, 2.44],
                                       [np.inf, 1.0, np.inf, np.inf, 2.78]),
                               max_nfev=5000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x
    return _params

def predict(row):
    u, s = _values([row])
    return float(_model(_params, u, s)[0])