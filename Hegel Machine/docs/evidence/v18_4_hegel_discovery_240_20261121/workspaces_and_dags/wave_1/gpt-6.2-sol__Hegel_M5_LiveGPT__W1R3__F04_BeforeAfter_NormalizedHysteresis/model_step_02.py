import math
import numpy as np
from scipy.optimize import least_squares

_scale = 0.004
_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    step = np.array([(r['u_now'] - r['u_past']) / _scale for r in rows])
    horizon = np.array([r['H'] for r in rows])
    target = np.array([r['y'] - r['x_now'] for r in rows])

    def residual(p):
        steady, amplitude, rate = p
        return step * (steady + amplitude * np.exp(-rate * horizon)) - target

    _params = least_squares(residual, [0.5, 1.5, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row):
    steady, amplitude, rate = _params
    step = (row['u_now'] - row['u_past']) / _scale
    return float(row['x_now'] + step * (steady + amplitude * math.exp(-rate * row['H'])))
