import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    adherence = math.exp(-z) / (1.0 + math.exp(-z)) if z >= 0 else 1.0 / (1.0 + math.exp(z))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * adherence
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    features = [_features(row) for row in train_rows]
    dose = np.array([x[0] for x in features])
    impact = np.array([x[1] for x in features])
    y = np.array([float(row['y']) for row in train_rows])
    def residual(p):
        g, k, c = p
        return g * dose / (k + dose) - c * impact - y
    result = least_squares(residual, [8.0, 1.0, 1.0], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    dose, impact = _features(row)
    g, k, c = _params
    return float(g * dose / (k + dose) - c * impact)
