import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 1.0
    s = float(row['s_grad'])
    a = u / (1.2 + u)
    return (a, a * s / (1.0 + s), math.sqrt(u))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([float(row['y']) for row in rows], dtype=float)

    def residual(p):
        v, rho, c = p
        return X @ np.array([v, -v * rho, c]) - y

    warm = least_squares(residual, [6.5, 0.70, 0.0], bounds=([3.836397, 0.658333, -0.822167], [9.163604, 0.748333, 1.108834]), gtol=1e-12)
    if np.sqrt(np.mean(warm.fun ** 2)) > 1e-6:
        wide = least_squares(residual, warm.x, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), gtol=1e-12)
        if wide.cost < warm.cost:
            warm = wide
    _params = warm.x
    return _params

def predict(row):
    v, rho, c = _params
    a, b, q = _features(row)
    return float(v * (a - rho * b) + c * q)
