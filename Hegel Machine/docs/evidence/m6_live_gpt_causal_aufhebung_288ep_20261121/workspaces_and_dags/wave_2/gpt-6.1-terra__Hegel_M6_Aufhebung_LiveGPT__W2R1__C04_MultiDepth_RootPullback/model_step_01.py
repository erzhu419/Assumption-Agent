import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    z = 6.0 * (float(row['a_foot']) - 1.5)
    adherence = math.exp(-z) / (1.0 + math.exp(-z)) if z >= 0 else 1.0 / (1.0 + math.exp(z))
    e = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor']) * adherence
    impact = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return e, impact

def fit(train_rows):
    global _params
    data = [(*_features(row), float(row['y'])) for row in train_rows]
    def residual(p):
        g, k, c = p
        return np.array([g * e / (k + e) - c * impact - y for e, impact, y in data])
    result = least_squares(residual, [8.0, 0.5, 1.0], bounds=([0.0, 1e-10, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    g, k, c = _params
    e, impact = _features(row)
    return float(g * e / (k + e) - c * impact)
