import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # tau_scale = 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    branch = c_loc * math.sqrt(max(0.0, u_n))
    avalanche = -k_aval * max(0.0, s - s_crit) * u_n
    return base + branch + avalanche

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    lo = np.array([1.0, 0.05, -2.0, 0.0, 0.0])
    hi = np.array([15.0, 0.95, 2.0, 10.0, 5.0])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    best = None
    for start in ([6.5, 0.70, 0.1, 1.0, 2.5], [7.0, 0.50, 0.5, 2.0, 3.0], [5.0, 0.85, -0.2, 0.5, 2.0]):
        result = least_squares(residual, np.clip(np.asarray(start, dtype=float), lo + 1e-8, hi - 1e-8), bounds=(lo, hi), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _params))