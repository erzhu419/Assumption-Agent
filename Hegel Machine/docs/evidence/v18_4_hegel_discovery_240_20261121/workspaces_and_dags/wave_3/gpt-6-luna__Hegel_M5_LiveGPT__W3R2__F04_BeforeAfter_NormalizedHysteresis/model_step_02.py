import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    step = float(row['u_now']) - float(row['u_past'])
    h = float(row['H'])
    x = float(row['x_now'])
    return x, step, h

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        amp, tau = p
        return X[:, 0] + amp * X[:, 1] * np.exp(-X[:, 2] / tau) - y
    result = least_squares(residual, x0=np.array([2.5, 0.7]), bounds=([-100.0, 1e-6], [100.0, 100.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp, tau = _params
    x, step, h = _features(row)
    return float(x + amp * step * np.exp(-h / tau))