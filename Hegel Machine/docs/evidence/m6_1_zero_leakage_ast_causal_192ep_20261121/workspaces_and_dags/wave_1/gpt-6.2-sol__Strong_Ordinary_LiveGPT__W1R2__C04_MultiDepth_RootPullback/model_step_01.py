import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _terms(row):
    indoor = float(row['u_indoor'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    t = math.exp(-abs(z))
    space = t / (1.0 + t) if z >= 0.0 else 1.0 / (1.0 + t)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * space
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.asarray([_terms(row) for row in rows], dtype=float).T
    y = np.asarray([float(row['y']) for row in rows])

    def residual(params):
        gain, half, cost = params
        return gain * dose / (half + dose) - cost * impact - y

    _params = least_squares(residual, [8.0, 1.5, 0.4], bounds=([0.0, 1e-8, 0.0], np.inf)).x

def predict(row) -> float:
    dose, impact = _terms(row)
    gain, half, cost = _params
    return float(gain * dose / (half + dose) - cost * impact)
