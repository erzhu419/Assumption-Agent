import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    step = np.array([r['u_now'] - r['u_past'] for r in rows], dtype=float)
    horizon = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        persistent, transient, rate = p
        return x + step * (persistent + transient * np.exp(-rate * horizon)) - y

    _params = least_squares(residual, [0.5, 2.0, 1.0], bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    persistent, transient, rate = _params
    step = row['u_now'] - row['u_past']
    return float(row['x_now'] + step * (persistent + transient * math.exp(-rate * row['H'])))
