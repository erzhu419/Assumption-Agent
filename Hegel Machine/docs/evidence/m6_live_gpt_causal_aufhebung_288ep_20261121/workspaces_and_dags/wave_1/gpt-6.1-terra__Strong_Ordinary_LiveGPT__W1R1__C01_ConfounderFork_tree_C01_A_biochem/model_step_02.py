import numpy as np
from scipy.optimize import least_squares

_params = None
_driver = None

def fit(train_rows):
    global _params, _driver
    y = np.array([r['y'] for r in train_rows], dtype=float)
    u = np.array([r['u_ctrl'] for r in train_rows], dtype=float)
    best = None
    for name in ['x_ind', 'z_1', 'z_2', 'z_3', 'z_4']:
        x = np.array([r[name] for r in train_rows], dtype=float)
        if name == 'x_ind':
            x = x / 1.5
        x2 = x * x
        def residual(p):
            return p[0] * x2 / (p[1] + x2) - p[2] * u - y
        result = least_squares(residual, [10.0, 1.0, 0.2], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
        loss = np.dot(result.fun, result.fun)
        if best is None or loss < best[0]:
            best = (loss, name, result.x)
    _, _driver, _params = best

def predict(row):
    x = float(row[_driver])
    if _driver == 'x_ind':
        x /= 1.5
    v, k, c = _params
    return float(v * x * x / (k + x * x) - c * float(row['u_ctrl']))