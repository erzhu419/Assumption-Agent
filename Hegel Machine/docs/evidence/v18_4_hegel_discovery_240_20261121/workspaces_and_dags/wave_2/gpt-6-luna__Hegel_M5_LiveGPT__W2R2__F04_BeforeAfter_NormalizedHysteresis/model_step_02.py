import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = train_rows
    delta = np.array([r['u_now'] - r['u_past'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        amp, tau = p
        return x + amp * delta * np.exp(-h / tau) - y
    res = least_squares(residual, x0=np.array([0.4, 1.5]), bounds=([0.0, 1e-6], [np.inf, np.inf]))
    _PARAMS = tuple(res.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    amp, tau = _PARAMS
    return float(row['x_now'] + amp * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / tau))