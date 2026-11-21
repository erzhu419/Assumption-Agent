import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, row):
        gain, tau_relax, k_eq = p
        un = float(row['u_now']) / 0.004
        up = float(row['u_past']) / 0.004
        H = float(row['H'])
        return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.6]), bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(p[0] * (float(row['u_now']) / 0.004 - float(row['u_past']) / 0.004) * math.exp(-float(row['H']) / p[1]) + p[2] * float(row['u_now']) / 0.004)
