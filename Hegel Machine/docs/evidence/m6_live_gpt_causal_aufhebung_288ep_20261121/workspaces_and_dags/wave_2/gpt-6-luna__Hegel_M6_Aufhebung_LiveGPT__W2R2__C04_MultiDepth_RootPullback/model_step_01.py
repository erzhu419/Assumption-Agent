import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u_out = float(row['u_outdoor'])
    u_in = float(row['u_indoor'])
    p = float(row['p_weather'])
    a = float(row['a_foot'])
    impact = float(row['impact_idx'])
    z = 6.0 * (a - 1.5)
    # Stable evaluation of the compact-space sigmoid denominator.
    if z >= 0:
        adherence = math.exp(-z) / (1.0 + math.exp(-z))
    else:
        adherence = 1.0 / (1.0 + math.exp(z))
    e_indoor = u_in * adherence
    e_eff = u_out * p + e_indoor
    loss_feature = (u_in * impact) ** 2
    return e_eff, loss_feature

def fit(train_rows):
    global _params
    features = [_features(row) for row in train_rows]
    y = np.asarray([float(row['y']) for row in train_rows], dtype=float)
    e = np.asarray([v[0] for v in features], dtype=float)
    loss = np.asarray([v[1] for v in features], dtype=float)

    def residual(theta):
        g_root, k_e, c_imp = theta
        return g_root * e / (k_e + e) - c_imp * loss - y

    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.5]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return {'g_root': _params[0], 'k_e': _params[1], 'c_imp': _params[2]}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    g_root, k_e, c_imp = _params
    e_eff, loss_feature = _features(row)
    return float(g_root * e_eff / (k_e + e_eff) - c_imp * loss_feature)
