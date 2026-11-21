import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    e_indoor = indoor / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    e_eff = float(row['u_outdoor']) * float(row['p_weather']) + e_indoor
    impact = (indoor * float(row['impact_idx'])) ** 2
    return e_eff, impact

def fit(train_rows):
    global _params
    features = np.array([_features(row) for row in train_rows], dtype=float)
    targets = np.array([float(row['y']) for row in train_rows])
    def residual(p):
        g, k, c = p
        return g * features[:, 0] / (k + features[:, 0]) - c * features[:, 1] - targets
    result = least_squares(residual, [8.0, 1.0, 1.0], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    e_eff, impact = _features(row)
    g, k, c = _params
    return float(g * e_eff / (k + e_eff) - c * impact)
