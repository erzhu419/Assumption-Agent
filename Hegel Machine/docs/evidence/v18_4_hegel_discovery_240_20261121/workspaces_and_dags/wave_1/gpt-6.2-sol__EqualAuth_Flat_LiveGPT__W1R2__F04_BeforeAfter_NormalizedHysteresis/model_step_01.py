import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU_SCALE = 1.0

def fit(train_rows):
    global _params
    rows = list(train_rows)
    now = np.array([float(r['u_now']) / _TAU_SCALE for r in rows])
    past = np.array([float(r['u_past']) / _TAU_SCALE for r in rows])
    horizon = np.array([float(r['H']) for r in rows])
    target = np.array([float(r['y']) for r in rows])
    delta = now - past

    def residual(p):
        gain, tau, k_eq = p
        return gain * delta * np.exp(-horizon / tau) + k_eq * now - target

    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, tau, k_eq = _params
    now = float(row['u_now']) / _TAU_SCALE
    past = float(row['u_past']) / _TAU_SCALE
    return float(gain * (now - past) * math.exp(-float(row['H']) / tau) + k_eq * now)
