import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    t = math.exp(-abs(z))
    adherence = (t if z >= 0.0 else 1.0) / (1.0 + t)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * adherence
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(p):
        g, k, c = p
        return g * dose / (k + dose) - c * impact - y

    _params = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], np.inf)).x

def predict(row) -> float:
    dose, impact = _features(row)
    g, k, c = _params
    return float(g * dose / (k + dose) - c * impact)
