import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return u, s

def _model(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    u, s = _values(train_rows)
    y = np.array [r['y'] for r in train_rows] if False else np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for k in (0.3, 1.0, 4.0):
        for a in (0.1, 1.0, 3.0):
            result = least_squares(lambda p: _model(p, u, s) - y, [8.0, k, a, 0.1], bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
    _params = best.x
    return _params

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_model(_params, u, s))