import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    un, up, h, y = np.asarray([(r['u_now']/TAU_SCALE, r['u_past']/TAU_SCALE, r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        gain, log_tau, k_eq = p
        return gain * (un - up) * np.exp(-h * np.exp(-log_tau)) + k_eq * un - y
    result = least_squares(residual, [1.0, math.log(float(np.median(h))), 1.0], bounds=([-np.inf, -20.0, -np.inf], [np.inf, 20.0, np.inf]))
    _params = result.x

def predict(row) -> float:
    gain, log_tau, k_eq = _params
    un = row['u_now'] / TAU_SCALE
    up = row['u_past'] / TAU_SCALE
    return float(gain * (un - up) * math.exp(-row['H'] * math.exp(-log_tau)) + k_eq * un)
