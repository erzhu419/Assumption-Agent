import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    gain, tau, keq = p
    scale = 0.004
    un = row['u_now'] / scale
    up = row['u_past'] / scale
    return gain * (un - up) * math.exp(-row['H'] / tau) + keq * un

def fit(train_rows):
    rows = train_rows
    def residual(p):
        return np.array([_pred(r, p) - r['y'] for r in rows])
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.0]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]))
    p = result.x
    return tuple(float(v) for v in p)

def predict(row):
    return float(_pred(row, fit(_TRAIN_ROWS)))