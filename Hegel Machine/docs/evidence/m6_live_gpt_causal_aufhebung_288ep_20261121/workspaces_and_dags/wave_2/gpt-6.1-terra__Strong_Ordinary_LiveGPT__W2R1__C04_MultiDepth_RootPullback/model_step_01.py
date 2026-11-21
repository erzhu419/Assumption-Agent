import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    if z >= 0:
        e = math.exp(-z)
        adherence = e / (1.0 + e)
    else:
        adherence = 1.0 / (1.0 + math.exp(z))
    indoor = float(row['u_indoor']) * adherence
    impact = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return outdoor + indoor, impact

def fit(train_rows):
    global _params
    features = np.array([_features(row) for row in train_rows], dtype=float)
    target = np.array([float(row['y']) for row in train_rows], dtype=float)
    dose, impact = features[:, 0], features[:, 1]
    def residual(p):
        g, k, c = p
        return g * dose / (k + dose) - c * impact - target
    result = least_squares(residual, [8.0, 0.8, 1.0], bounds=([0.0, 1e-10, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    dose, impact = _features(row)
    g, k, c = _params
    return float(g * dose / (k + dose) - c * impact)
