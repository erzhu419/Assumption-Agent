import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    step = np.array([100.0 * (r['u_now'] - r['u_past']) for r in rows])
    horizon = np.array([r['H'] for r in rows], dtype=float)
    shift = np.array([r['y'] - r['x_now'] for r in rows])

    def residual(p):
        a, b, k = p
        return step * (a + b * np.exp(-k * horizon)) - shift

    _params = least_squares(residual, [1.5, 4.0, 1.3], bounds=(0.0, np.inf)).x

def predict(row):
    a, b, k = _params
    step = 100.0 * (row['u_now'] - row['u_past'])
    return float(row['x_now'] + step * (a + b * math.exp(-k * row['H'])))