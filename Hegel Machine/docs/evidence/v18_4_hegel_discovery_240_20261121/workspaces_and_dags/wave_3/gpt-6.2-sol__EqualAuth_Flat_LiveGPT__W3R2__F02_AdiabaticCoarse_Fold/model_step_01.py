import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.array([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        gain, rate = p
        equilibrium = gain * u
        return equilibrium + (x - equilibrium) * np.exp(-rate * h) - y
    gain0 = np.linalg.lstsq(u[:, None], y, rcond=None)[0][0]
    _params = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row):
    gain, rate = _params
    equilibrium = gain * row['u_now']
    return float(equilibrium + (row['x_now'] - equilibrium) * math.exp(-rate * row['H']))
