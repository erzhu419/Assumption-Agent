import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    scale = 0.004
    un = float(row['u_now']) / scale
    up = float(row['u_past']) / scale
    h = float(row['H'])
    return un, up, h

def _pred(row, p):
    gain, tau_relax, k_eq = p
    un, up, h = _features(row)
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        tau = max(1e-8, float(p[1]))
        pp = (float(p[0]), tau, float(p[2]))
        return np.asarray([_pred(r, pp) for r in rows]) - y
    result = least_squares(residual, x0=np.array([-1.0, 0.6, 0.46]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    return (float(result.x[0]), max(1e-8, float(result.x[1])), float(result.x[2]))

def predict(row):
    return float(_pred(row, fit(_TRAIN_ROWS)))