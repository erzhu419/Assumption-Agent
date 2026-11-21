import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        k, a = p
        return k*u + (x-k*u)*np.exp(-a*h) - y
    candidates = [least_squares(residual, [2.35, a], bounds=([-100.0, 0.0], [100.0, 100.0])) for a in (0.5, 1.0, 2.0)]
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    k, a = _params
    u, x, h = float(row['u_now']), float(row['x_now']), float(row['H'])
    return float(k*u + (x-k*u)*np.exp(-a*h))