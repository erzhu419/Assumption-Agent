import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    step = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    target = np.array([r['y'] - r['x_now'] for r in train_rows], dtype=float)
    def residual(p):
        gain, amplitude, rate = p
        return step * (gain - amplitude * np.exp(-rate * h)) - target
    _params = least_squares(residual, [80.0, 400.0, 1.0], bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]), ftol=1e-13, xtol=1e-13, gtol=1e-13).x

def predict(row):
    gain, amplitude, rate = _params
    step = row['u_now'] - row['u_past']
    return float(row['x_now'] + step * (gain - amplitude * math.exp(-rate * row['H'])))