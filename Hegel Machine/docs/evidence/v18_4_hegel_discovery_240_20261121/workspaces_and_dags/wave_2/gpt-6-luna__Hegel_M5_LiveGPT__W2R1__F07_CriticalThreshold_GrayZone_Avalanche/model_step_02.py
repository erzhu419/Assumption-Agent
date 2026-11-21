import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _formula(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(max(u, 0.0))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_formula(r, p) for r in train_rows]) - y
    starts = [
        [6.5, 0.70, 0.1, 1.5, 2.6],
        [5.0, 0.70, 0.5, 2.0, 2.5],
        [8.0, 0.68, -0.2, 1.0, 2.7],
        [6.0, 0.72, 0.8, 2.5, 2.4],
    ]
    lo = [0.01, 0.0, -5.0, 0.0, 2.0]
    hi = [30.0, 0.99, 5.0, 10.0, 3.5]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _params = best[1]
    return {'vmax': float(_params[0]), 'rho': float(_params[1]),
            'c_loc': float(_params[2]), 'k_aval': float(_params[3]),
            's_crit': float(_params[4]), 'train_rmse': math.sqrt(best[0] / len(y))}

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_formula(row, _params))