import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def fit(train_rows):
    global _fit_params
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u + p[1] - y
    result = least_squares(residual, np.array([1.95, 0.0]))
    _fit_params = result.x
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_fit_params[0] * float(row['u_now']) + _fit_params[1])