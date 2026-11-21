import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(float(row['u_now']), 1e-8)
    return np.array([u, float(row['s_grad']) - 1.0, float(row['x_now']) - 0.25,
                     float(row['mem_conv']), float(row['u_past']) - 1.0,
                     float(row['H']) - 0.6], dtype=float)

def _model(theta, row):
    a, b, p, g, x, m, past, h = theta
    f = _features(row)
    return b + a * (f[0] ** p) + g*f[1] + x*f[2] + m*f[3] + past*f[4] + h*f[5]

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(theta):
        return np.array([_model(theta, r) for r in rows]) - y
    best = None
    for p0 in (0.25, 0.5, 0.8, 1.2):
        theta0 = np.array([1.5, 0.0, p0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = least_squares(residual, theta0,
            bounds=([0.0, -100.0, 0.05, -100.0, -100.0, -100.0, -100.0, -100.0],
                    [100.0, 100.0, 2.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
            max_nfev=5000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_PARAMS, row))