import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, target = data.T
    now = now / _SCALE
    step = now - past / _SCALE
    def residual(p):
        gain, rate, equilibrium = p
        return gain * step * np.exp(-rate * horizon) + equilibrium * now - target
    _params = least_squares(residual, [-1.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, equilibrium = _params
    now = row['u_now'] / _SCALE
    step = now - row['u_past'] / _SCALE
    return float(gain * step * math.exp(-rate * row['H']) + equilibrium * now)
