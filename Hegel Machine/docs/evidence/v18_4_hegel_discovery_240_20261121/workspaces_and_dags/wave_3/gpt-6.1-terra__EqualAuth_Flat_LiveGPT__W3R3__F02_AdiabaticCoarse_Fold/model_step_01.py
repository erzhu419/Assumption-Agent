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
    k0 = np.dot(u, y) / np.dot(u, u)
    result = least_squares(residual, [k0, 1.5], bounds=([-10.0, 0.0], [10.0, 20.0]))
    _params = result.x

def predict(row):
    k, a = _params
    u, x, h = row['u_now'], row['x_now'], row['H']
    return float(k*u + (x-k*u)*np.exp(-a*h))
