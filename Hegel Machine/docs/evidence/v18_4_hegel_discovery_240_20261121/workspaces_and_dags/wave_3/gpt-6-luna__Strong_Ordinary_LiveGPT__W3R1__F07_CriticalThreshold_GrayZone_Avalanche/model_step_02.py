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
    branch = c_loc * math.sqrt(max(u_n, 0.0))
    avalanche = k_aval * max(0.0, s - s_crit) * u_n
    return base + branch - avalanche

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    best = None
    for p0 in ([5.0, 0.7, 0.0, 1.0, 2.6], [4.0, 0.5, 0.2, 0.5, 2.5], [8.0, 1.0, -0.2, 2.0, 2.8]):
        result = least_squares(residual, p0,
            bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 3.0, 10.0, 20.0, 3.2]),
            max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))