import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    past, now, x, h, y = np.asarray([[r[k] for k in ('u_past', 'u_now', 'x_now', 'H', 'y')] for r in train_rows], dtype=float).T
    step = now - past

    def residual(p):
        gain, kick, rate = p
        equilibrium = gain * now
        return equilibrium + (x - equilibrium + kick * step) * np.exp(-rate * h) - y

    _params = least_squares(residual, [0.6, 2.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    gain, kick, rate = _params
    now = row['u_now']
    equilibrium = gain * now
    step = now - row['u_past']
    return float(equilibrium + (row['x_now'] - equilibrium + kick * step) * math.exp(-rate * row['H']))
