import math
import numpy as np
from scipy.optimize import least_squares

SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    now = np.array([r['u_now'] / SCALE for r in train_rows], dtype=float)
    step = np.array([(r['u_now'] - r['u_past']) / SCALE for r in train_rows], dtype=float)
    horizon = np.array([r['H'] for r in train_rows], dtype=float)
    target = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        gain, log_tau, equilibrium = p
        return equilibrium * now + gain * step * np.exp(-horizon / np.exp(log_tau)) - target

    result = least_squares(residual, [-1.0, math.log(0.65), 0.0], bounds=([-np.inf, -15.0, -np.inf], [np.inf, 15.0, np.inf]))
    _params = tuple(map(float, result.x))
    return _params

def predict(row):
    gain, log_tau, equilibrium = _params
    now = row['u_now'] / SCALE
    step = (row['u_now'] - row['u_past']) / SCALE
    return float(equilibrium * now + gain * step * math.exp(-row['H'] / math.exp(log_tau)))