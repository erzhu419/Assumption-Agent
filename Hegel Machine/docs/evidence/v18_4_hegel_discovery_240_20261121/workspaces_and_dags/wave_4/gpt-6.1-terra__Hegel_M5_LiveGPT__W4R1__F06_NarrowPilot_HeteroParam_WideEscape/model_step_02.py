import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, rows):
    v, rho, c = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return v * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    y = np.array [r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _model(p, train_rows) - y, [7.0, 0.5, 0.2], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_model(_params, [row])[0])