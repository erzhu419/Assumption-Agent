import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def fit(train_rows):
    global _params
    past, now, horizon, y = np.asarray([
        (r['u_past'] / TAU_SCALE, r['u_now'] / TAU_SCALE, r['H'], r['y'])
        for r in train_rows
    ], dtype=float).T
    step = now - past
    gain0, eq0 = np.linalg.lstsq(
        np.column_stack((step * np.exp(-horizon / TAU_SCALE), now)), y, rcond=None
    )[0]

    def residual(p):
        gain, tau, eq = p
        return gain * step * np.exp(-horizon / tau) + eq * now - y

    _params = least_squares(residual, [gain0, TAU_SCALE, eq0],
                           bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    gain, tau, eq = _params
    past = row['u_past'] / TAU_SCALE
    now = row['u_now'] / TAU_SCALE
    return float(gain * (now - past) * math.exp(-row['H'] / tau) + eq * now)
