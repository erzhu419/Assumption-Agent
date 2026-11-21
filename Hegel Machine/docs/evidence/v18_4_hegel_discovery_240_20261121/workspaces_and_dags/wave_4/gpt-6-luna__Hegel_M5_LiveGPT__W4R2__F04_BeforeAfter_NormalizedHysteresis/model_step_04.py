import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    gain, tau_relax, k_eq = p
    un = float(row['u_now']) / 0.004
    up = float(row['u_past']) / 0.004
    h = float(row['H'])
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        return np.array([_pred(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([-1.0, 0.5, 0.3]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit([row]) if False else _MODEL_PARAMS
    return float(_pred(p, row))
