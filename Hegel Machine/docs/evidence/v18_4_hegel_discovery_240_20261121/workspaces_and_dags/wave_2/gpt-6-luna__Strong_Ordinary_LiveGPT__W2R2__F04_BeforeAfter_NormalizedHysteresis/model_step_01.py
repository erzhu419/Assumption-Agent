import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    gain, tau_relax, k_eq = p
    scale = 5.0
    un = float(row['u_now']) / scale
    up = float(row['u_past']) / scale
    H = float(row['H'])
    return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        return np.array([_pred(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([1.0, 5.0, 1.0]), bounds=([-20.0, 0.05, -20.0], [20.0, 100.0, 20.0]))
    return result.x

def predict(row):
    p = fit([row])
    return float(_pred(row, p))