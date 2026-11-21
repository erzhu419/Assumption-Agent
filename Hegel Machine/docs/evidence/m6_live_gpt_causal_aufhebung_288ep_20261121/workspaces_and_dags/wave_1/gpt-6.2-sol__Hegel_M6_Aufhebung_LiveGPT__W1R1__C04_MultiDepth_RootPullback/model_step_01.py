import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    adherence = math.exp(-float(np.logaddexp(0.0, z)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * adherence
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([float(row['y']) for row in rows])

    def residuals(params):
        g, k, c = params
        return g * dose / (k + dose) - c * impact - y

    initial = [max(1.0, 1.5 * float(y.max())), 1.0, 0.5]
    _params = least_squares(residuals, initial, bounds=([0.0, 1e-9, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    g, k, c = _params
    dose, impact = _features(row)
    return float(g * dose / (k + dose) - c * impact)
