import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    z = np.clip(6.0 * (float(row['a_foot']) - 1.5), -700.0, 700.0)
    indoor = float(row['u_indoor'])
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor / (1.0 + np.exp(z))
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    features = np.array([_features(row) for row in train_rows], dtype=float)
    y = np.array([float(row['y']) for row in train_rows])
    dose, impact = features[:, 0], features[:, 1]
    def residual(p):
        g, k, c = p
        return g * dose / (k + dose) - c * impact - y
    result = least_squares(residual, [8.0, 1.0, 1.0], bounds=([0.0, 1e-10, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row) -> float:
    dose, impact = _features(row)
    g, k, c = _params
    return float(g * dose / (k + dose) - c * impact)
