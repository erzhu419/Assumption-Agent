import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    indoor = float(row['u_indoor']) / (1.0 + math.exp(max(-700.0, min(700.0, z))))
    impact = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return outdoor + indoor, impact

def fit(train_rows):
    global _params
    def residual(params):
        g_root, k_e, c_imp = params
        return np.asarray([
            g_root * (dose / (k_e + dose)) - c_imp * impact - float(row['y'])
            for row in train_rows
            for dose, impact in [_features(row)]
        ])
    result = least_squares(residual, x0=[8.0, 1.0, 1.0], bounds=([0.0, 1e-8, 0.0], ([100.0, 100.0, 100.0])))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    g_root, k_e, c_imp = _params
    dose, impact = _features(row)
    return float(g_root * dose / (k_e + dose) - c_imp * impact)
