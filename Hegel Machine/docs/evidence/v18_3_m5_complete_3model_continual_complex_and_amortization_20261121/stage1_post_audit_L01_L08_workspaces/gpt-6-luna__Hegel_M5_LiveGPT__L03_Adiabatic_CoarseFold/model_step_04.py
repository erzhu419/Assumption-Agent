import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        c0, beta = p
        return c0 * u * (1.0 + beta * np.cbrt(H)) - y
    result = least_squares(residual, np.array([1.0, 1.0]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    c0, beta = _params
    return float(c0 * row['u_now'] * (1.0 + beta * np.cbrt(row['H'])))