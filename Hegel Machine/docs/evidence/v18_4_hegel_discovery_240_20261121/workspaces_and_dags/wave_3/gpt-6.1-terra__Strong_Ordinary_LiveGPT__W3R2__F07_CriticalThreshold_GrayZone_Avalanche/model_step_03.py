import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_evaluate(r, p) for r in rows]) - y
    best = None
    for threshold in (2.48, 2.55, 2.62, 2.68):
        for rho in (0.5, 0.7, 0.9):
            result = least_squares(residual, [8.0, rho, 0.5, 2.0, threshold],
                                   bounds=([0.0, 0.0, -20.0, 0.0, 2.44],
                                           [100.0, 2.0, 20.0, 50.0, 2.78]),
                                   max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(row, _params))
