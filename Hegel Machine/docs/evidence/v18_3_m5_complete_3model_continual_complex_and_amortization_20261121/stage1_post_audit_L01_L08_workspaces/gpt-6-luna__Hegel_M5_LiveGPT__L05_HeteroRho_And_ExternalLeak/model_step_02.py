import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    base = (1.0 - _params[1] * s / (1.0 + s)) * (u / (1.2 + u))
    return base, u, np.sqrt(max(x, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        pred = p[0] * (1.0 - p[1] * np.asarray([r['s_grad'] for r in train_rows]) / (1.0 + np.asarray([r['s_grad'] for r in train_rows]))) * (np.asarray([r['u_now'] for r in train_rows]) / (1.2 + np.asarray([r['u_now'] for r in train_rows]))) - p[2] * np.asarray([r['u_now'] for r in train_rows]) + p[3] * np.sqrt(np.maximum(np.asarray([r['x_now'] for r in train_rows]), 0.0))
        return pred - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.03, 0.1]), bounds=([1.0, 0.05, -2.0, -2.0], [15.0, 0.95, 2.0, 2.0]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    return float(_params[0] * (1.0 - _params[1] * s / (1.0 + s)) * (u / (1.2 + u)) - _params[2] * u + _params[3] * np.sqrt(max(x, 0.0)))