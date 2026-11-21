import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    pw = float(row['p_weather'])
    af = float(row['a_foot'])
    impact = float(row['impact_idx'])
    # Clip exponent input to avoid overflow for unusual footprints.
    z = max(-700.0, min(700.0, 6.0 * (af - 1.5)))
    e_indoor = ui / (1.0 + math.exp(z))
    e_eff = uo * pw + e_indoor
    loss = (ui * impact) ** 2
    return e_eff, loss

def fit(train_rows):
    global _params
    x = [_features(r) for r in train_rows]
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(theta):
        g_root, k_e, c_imp = theta
        pred = np.asarray([g_root * (e / (k_e + e)) - c_imp * loss for e, loss in x])
        return pred - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.5]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    g_root, k_e, c_imp = _params
    e_eff, loss = _features(row)
    return float(g_root * (e_eff / (k_e + e_eff)) - c_imp * loss)