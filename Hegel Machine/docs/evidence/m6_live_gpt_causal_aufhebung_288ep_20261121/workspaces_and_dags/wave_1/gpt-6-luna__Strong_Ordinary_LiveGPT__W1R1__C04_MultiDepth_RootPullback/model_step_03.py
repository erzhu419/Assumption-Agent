import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def prediction(row, p):
        g_root, k_e, c_imp = p
        indoor = row['u_indoor'] / (1.0 + math.exp(6.0 * (row['a_foot'] - 1.5)))
        dose = row['u_outdoor'] * row['p_weather'] + indoor
        loss = c_imp * (row['u_indoor'] * row['impact_idx']) ** 2
        return g_root * dose / (k_e + dose) - loss
    def residuals(p):
        return np.array([prediction(r, p) - r['y'] for r in rows])
    result = least_squares(residuals, x0=np.array([8.0, 1.0, 1.0]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    g_root, k_e, c_imp = _PARAMS
    indoor = row['u_indoor'] / (1.0 + math.exp(6.0 * (row['a_foot'] - 1.5)))
    dose = row['u_outdoor'] * row['p_weather'] + indoor
    loss = c_imp * (row['u_indoor'] * row['impact_idx']) ** 2
    return float(g_root * dose / (k_e + dose) - loss)