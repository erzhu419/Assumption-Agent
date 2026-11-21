import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u_n = float(row['u_now']) / 0.1
    s = float(row['s_grad'])
    hill = u_n**2
    return u_n, hill, s

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def pred(p, x):
        u, h, s = x.T
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * h / (k_half + h) / (1.0 + alpha_inh * s) - c_leak * u
    def residual(p):
        return pred(p, X) - y
    p0 = np.array([6.0, 1.0, 0.5, 0.1])
    result = least_squares(residual, p0, bounds=([0.0, 1e-8, 0.0, 0.0], [30.0, 100.0, 20.0, 10.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    u, h, s = _features(row)
    vmax, k_half, alpha_inh, c_leak = _PARAMS
    return float(vmax * h / (k_half + h) / (1.0 + alpha_inh * s) - c_leak * u)
