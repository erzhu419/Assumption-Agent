import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, params):
    g_root, k_e, c_imp = params
    u_outdoor = float(row['u_outdoor'])
    u_indoor = float(row['u_indoor'])
    p_weather = float(row['p_weather'])
    a_foot = float(row['a_foot'])
    impact_idx = float(row['impact_idx'])
    e_indoor = u_indoor / (1.0 + math.exp(6.0 * (a_foot - 1.5)))
    e_eff = u_outdoor * p_weather + e_indoor
    return g_root * (e_eff / (k_e + e_eff)) - c_imp * ((u_indoor * impact_idx) ** 2)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residuals(params):
        return np.asarray([_predict_with(r, params) for r in rows]) - y
    result = least_squares(residuals, x0=np.array([8.0, 1.0, 1.0]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(row, _PARAMS))