import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 1.0
_params = None

def fit(train_rows):
    global _params
    up, un, h, y = np.asarray([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    up = up / _TAU_SCALE
    un = un / _TAU_SCALE
    delta = un - up

    def residual(p):
        gain, log_tau, k_eq = p
        return gain * delta * np.exp(-h * np.exp(-log_tau)) + k_eq * un - y

    _params = least_squares(residual, [1.0, 0.0, 0.5]).x

def predict(row) -> float:
    gain, log_tau, k_eq = _params
    un = row['u_now'] / _TAU_SCALE
    up = row['u_past'] / _TAU_SCALE
    return float(gain * (un - up) * math.exp(-row['H'] * math.exp(-log_tau)) + k_eq * un)
