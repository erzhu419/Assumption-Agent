import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _evaluate(p, u, s, root_u):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * root_u - k_aval * u * np.maximum(0.0, s - s_crit)


def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    result = least_squares(
        lambda p: _evaluate(p, u, s, root_u) - y,
        [5.0, 0.7, 0.5, 1.0, 2.6],
        bounds=([0.0, 0.0, -np.inf, 0.0, 2.44],
                [np.inf, 1.0, np.inf, np.inf, 2.78]),
    )
    _params = result.x
    return _params


def predict(row) -> float:
    u = float(row['u_now'])
    return float(_evaluate(_params, u, float(row['s_grad']), math.sqrt(u)))