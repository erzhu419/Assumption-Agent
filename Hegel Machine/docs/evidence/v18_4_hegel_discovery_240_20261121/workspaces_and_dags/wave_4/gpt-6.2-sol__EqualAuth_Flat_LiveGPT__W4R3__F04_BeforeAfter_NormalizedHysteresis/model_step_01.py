import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_past'], r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float)
    past, now, h, y = data.T
    now = now / _SCALE
    step = now - past / _SCALE
    tau0 = float(np.median(h))
    basis = np.column_stack((step * np.exp(-h / tau0), now))
    gain0, eq0 = np.linalg.lstsq(basis, y, rcond=None)[0]

    def residual(p):
        gain, tau, eq = p
        return gain * step * np.exp(-h / tau) + eq * now - y

    _params = least_squares(residual, [gain0, tau0, eq0], bounds=([-np.inf, 1e-10, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row):
    gain, tau, eq = _params
    now = row['u_now'] / _SCALE
    step = now - row['u_past'] / _SCALE
    return float(gain * step * math.exp(-row['H'] / tau) + eq * now)
