import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[(r['u_now'] - r['u_past']) / 0.004, r['H'], r['y'] - r['x_now']] for r in train_rows], dtype=float)
    step, horizon, target = data.T

    def residual(p):
        steady, transient, rate = p
        return step * (steady + transient * np.exp(-rate * horizon)) - target

    _params = least_squares(residual, [0.0, -1.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    steady, transient, rate = _params
    step = (row['u_now'] - row['u_past']) / 0.004
    return float(row['x_now'] + step * (steady + transient * math.exp(-rate * row['H'])))
