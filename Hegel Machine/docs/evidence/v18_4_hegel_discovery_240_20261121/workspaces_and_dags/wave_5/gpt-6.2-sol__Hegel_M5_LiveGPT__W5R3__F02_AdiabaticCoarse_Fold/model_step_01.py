import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r[k] for k in ('u_now', 'x_now', 'H', 'y')] for r in train_rows], dtype=float).T
    gain0 = float(np.dot(u, y) / np.dot(u, u))

    def residual(p):
        gain, rate = p
        equilibrium = gain * u
        return equilibrium + (x - equilibrium) * np.exp(-rate * h) - y

    result = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _params = tuple(map(float, result.x))
    return _params

def predict(row):
    gain, rate = _params
    equilibrium = gain * row['u_now']
    return float(equilibrium + (row['x_now'] - equilibrium) * math.exp(-rate * row['H']))
