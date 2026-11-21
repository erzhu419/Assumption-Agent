import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    x = np.asarray([r['x_now'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        k, b, a = p
        equilibrium = k * u + b
        return equilibrium + (x - equilibrium) * np.exp(-a * h) - y
    result = least_squares(residual, [2.35, 0.0, 1.5], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 20.0]))
    _params = result.x

def predict(row):
    k, b, a = _params
    equilibrium = k * float(row['u_now']) + b
    return float(equilibrium + (float(row['x_now']) - equilibrium) * np.exp(-a * float(row['H'])))
