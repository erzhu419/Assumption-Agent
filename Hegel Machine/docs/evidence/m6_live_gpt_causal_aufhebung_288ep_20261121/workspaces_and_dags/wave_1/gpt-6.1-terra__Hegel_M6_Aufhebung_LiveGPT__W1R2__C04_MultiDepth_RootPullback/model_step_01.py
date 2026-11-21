import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    z = 6.0 * (float(row['a_foot']) - 1.5)
    adherence = math.exp(-z) / (1.0 + math.exp(-z)) if z >= 0 else 1.0 / (1.0 + math.exp(z))
    indoor = float(row['u_indoor'])
    effective = float(row['u_outdoor']) * float(row['p_weather']) + indoor * adherence
    impact = (indoor * float(row['impact_idx'])) ** 2
    return effective, impact

def fit(train_rows):
    global _params
    features = [_features(row) for row in train_rows]
    targets = np.asarray([float(row['y']) for row in train_rows])
    def residual(params):
        g, k, c = params
        return np.asarray([g * e / (k + e) - c * impact for e, impact in features]) - targets
    result = least_squares(residual, [6.0, 0.5, 1.0], bounds=([0.0, 1e-10, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit before predict')
    e, impact = _features(row)
    g, k, c = _params
    return float(g * e / (k + e) - c * impact)
