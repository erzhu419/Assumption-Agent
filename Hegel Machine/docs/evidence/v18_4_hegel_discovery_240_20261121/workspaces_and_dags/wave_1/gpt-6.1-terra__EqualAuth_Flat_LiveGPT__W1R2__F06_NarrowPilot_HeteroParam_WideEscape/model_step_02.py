import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # The reciprocal gradient factor is approximately linear across the
    # pilot and both open-branch observations.
    def residual(p):
        a, km, b = p
        return a * u / (km + u) / (1.0 + b * s) - y
    result = least_squares(residual, [8.0, 1.2, 1.0], bounds=([0.0, 1e-6, 0.0], [100.0, 100.0, 100.0]))
    _params = result.x
    return _params

def predict(row):
    a, km, b = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(a * u / (km + u) / (1.0 + b * s))
