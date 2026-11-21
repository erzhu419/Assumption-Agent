import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, root_u, p):
    vmax, sink, c_loc, k_aval, s_crit = p
    return (vmax - sink * s / (1.0 + s)) * u / (1.2 + u) + c_loc * root_u - k_aval * u * np.maximum(0.0, s - s_crit)

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    result = least_squares(lambda p: _law(u, s, root_u, p) - y,
                           [7.0, 4.5, 0.4, 1.5, 2.6],
                           bounds=([0, 0, 0, 0, 2.44], [np.inf, np.inf, np.inf, np.inf, 2.78]))
    _params = result.x

def predict(row) -> float:
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_law(u, s, math.sqrt(u), _params))