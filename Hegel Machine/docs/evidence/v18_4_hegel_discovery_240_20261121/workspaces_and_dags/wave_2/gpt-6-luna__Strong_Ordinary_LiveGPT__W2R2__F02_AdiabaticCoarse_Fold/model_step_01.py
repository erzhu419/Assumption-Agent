import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now'])
    u_p = float(row['u_past'])
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 1.0, 1.8]), bounds=([-100.0, 1e-4, -100.0], [100.0, 100.0, 100.0]))
    p = result.x
    return {'gain': float(p[0]), 'tau_relax': float(p[1]), 'k_eq': float(p[2])}

def predict(row):
    p = fit([dict(row, y=0.0)]) if False else None
    raise RuntimeError('predict requires parameters returned by fit; use predict(row) with fitted model state')
