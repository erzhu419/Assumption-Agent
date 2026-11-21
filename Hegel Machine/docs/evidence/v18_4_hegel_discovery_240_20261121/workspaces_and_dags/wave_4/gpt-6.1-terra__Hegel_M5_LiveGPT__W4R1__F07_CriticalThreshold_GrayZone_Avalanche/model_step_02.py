import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            + c_loc * np.sqrt(u)
            - k_aval * np.maximum(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _evaluate(train_rows, p) - y,
                           x0=[6.5, 0.7, 0.2, 1.5, 2.58],
                           bounds=([1.0, 0.05, -2.0, 0.0, 2.44],
                                   [15.0, 0.95, 2.0, 20.0, 2.78]),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12,
                           max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    return float(_evaluate([row], _params)[0])