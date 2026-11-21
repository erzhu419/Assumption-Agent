import math
import numpy as np
from scipy.optimize import least_squares

_scale = 0.02
_params = None


def fit(train_rows):
    global _params
    data = np.array([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, y = data.T
    step = (now - past) / _scale
    current = now / _scale

    def residual(p):
        gain, rate, equilibrium = p
        return gain * step * np.exp(-rate * horizon) + equilibrium * current - y

    _params = least_squares(residual, [1.0, 1.0, 0.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x


def predict(row) -> float:
    gain, rate, equilibrium = _params
    step = (row['u_now'] - row['u_past']) / _scale
    current = row['u_now'] / _scale
    return float(gain * step * math.exp(-rate * row['H']) + equilibrium * current)
