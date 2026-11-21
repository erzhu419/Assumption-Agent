import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    space = 0.5 * (1.0 - math.tanh(3.0 * (float(row['a_foot']) - 1.5)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * space
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.array([_features(row) for row in rows], dtype=float).T
    y = np.array([float(row['y']) for row in rows])
    def residual(params):
        g, k, c = params
        return g * dose / (k + dose) - c * impact - y
    _params = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    g, k, c = _params
    dose, impact = _features(row)
    return float(g * dose / (k + dose) - c * impact)
