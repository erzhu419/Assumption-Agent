import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad']) - 1.0
    return u, g

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    z = np.asarray([_features(r) for r in train_rows], dtype=float)
    u, g = z[:, 0], z[:, 1]
    def residual(p):
        a, b, k, c0, c1 = p
        pred = a + b * (1.0 - np.exp(-k * u)) + g * (c0 + c1 * u)
        return pred - y
    result = least_squares(residual, x0=np.array([0.5, 4.5, 0.7, 0.0, 0.0]), bounds=([-20, -20, 1e-5, -20, -20], [20, 20, 20, 20, 20]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u, g = _features(row)
    a, b, k, c0, c1 = p
    return float(a + b * (1.0 - np.exp(-k * u)) + g * (c0 + c1 * u))

_TRAIN_ROWS = []

def fit(train_rows):
    global _TRAIN_ROWS
    _TRAIN_ROWS = list(train_rows)
    y = np.asarray([float(r['y']) for r in _TRAIN_ROWS], dtype=float)
    z = np.asarray([_features(r) for r in _TRAIN_ROWS], dtype=float)
    u, g = z[:, 0], z[:, 1]
    def residual(p):
        a, b, k, c0, c1 = p
        return a + b * (1.0 - np.exp(-k * u)) + g * (c0 + c1 * u) - y
    result = least_squares(residual, x0=np.array([0.5, 4.5, 0.7, 0.0, 0.0]), bounds=([-20, -20, 1e-5, -20, -20], [20, 20, 20, 20, 20]))
    return result.x

def predict(row):
    a, b, k, c0, c1 = fit(_TRAIN_ROWS)
    u, g = _features(row)
    return float(a + b * (1.0 - np.exp(-k * u)) + g * (c0 + c1 * u))