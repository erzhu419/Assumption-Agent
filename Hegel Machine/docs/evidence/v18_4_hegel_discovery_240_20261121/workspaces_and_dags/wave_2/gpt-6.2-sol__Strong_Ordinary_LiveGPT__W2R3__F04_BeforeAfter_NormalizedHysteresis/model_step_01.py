import math
import numpy as np
from scipy.optimize import least_squares

SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    a = np.array([[r['u_now'] / SCALE, (r['u_now'] - r['u_past']) / SCALE, r['H'], r['y']] for r in train_rows], dtype=float)
    u, step, h, y = a.T
    def residual(p):
        gain, log_tau, k_eq = p
        return k_eq * u + gain * step * np.exp(-h / np.exp(log_tau)) - y
    gain0, k0 = np.linalg.lstsq(np.column_stack((step * np.exp(-h), u)), y, rcond=None)[0]
    _params = least_squares(residual, [gain0, 0.0, k0]).x

def predict(row) -> float:
    gain, log_tau, k_eq = _params
    u = row['u_now'] / SCALE
    step = (row['u_now'] - row['u_past']) / SCALE
    return float(k_eq * u + gain * step * math.exp(-row['H'] / math.exp(log_tau)))
