import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    outdoor = float(row['u_outdoor']) * float(row['p_weather'])
    indoor = float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    impact = (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return outdoor + indoor, impact

def fit(train_rows):
    global _PARAMS
    doses, impacts, ys = [], [], []
    for row in train_rows:
        dose, impact = _features(row)
        doses.append(dose)
        impacts.append(impact)
        ys.append(float(row['y']))
    doses = np.asarray(doses, dtype=float)
    impacts = np.asarray(impacts, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # y = g_root * E_eff/(k_e + E_eff) - c_imp * impact
    def residuals(params):
        g_root, k_e, c_imp = params
        return g_root * doses / (k_e + doses) - c_imp * impacts - ys

    result = least_squares(residuals, x0=np.array([8.0, 1.0, 1.0]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    g_root, k_e, c_imp = _PARAMS
    dose, impact = _features(row)
    return float(g_root * dose / (k_e + dose) - c_imp * impact)