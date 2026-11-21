import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    sat = u / (1.2 + u)
    grad = s / (1.0 + s)
    local = np.sqrt(u)

    def residual(p):
        vmax, sink, c_loc, k_aval, s_crit = p
        return (vmax - sink * grad) * sat + c_loc * local - k_aval * u * np.maximum(0.0, s - s_crit) - y

    _params = least_squares(residual, [6.0, 3.0, 0.5, 1.0, 2.59], bounds=([0.0, 0.0, 0.0, 0.0, 2.50], [np.inf, np.inf, np.inf, np.inf, 2.68])).x

def predict(row):
    vmax, sink, c_loc, k_aval, s_crit = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float((vmax - sink * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u) - k_aval * u * max(0.0, s - s_crit))