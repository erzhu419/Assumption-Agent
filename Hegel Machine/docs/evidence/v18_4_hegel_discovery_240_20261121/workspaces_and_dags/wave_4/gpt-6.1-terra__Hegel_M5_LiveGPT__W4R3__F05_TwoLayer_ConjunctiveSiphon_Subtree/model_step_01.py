import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, s / (1.0 + s), np.sqrt(u)

def fit(train_rows):
    global _params
    features = [_features(row) for row in train_rows]
    y = np.array([row['y'] for row in train_rows], dtype=float)
    def residual(p):
        v, rho, c = p
        return np.array([v * (1.0 - rho * s) * b + c * q for b, s, q in features]) - y
    result = least_squares(residual, [6.5, 0.70, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    b, s, q = _features(row)
    v, rho, c = _params
    return float(v * (1.0 - rho * s) * b + c * q)
