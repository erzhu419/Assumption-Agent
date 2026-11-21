import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p, branch_sign):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic_tau_scale = 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + branch_sign * c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    # The archived mechanism leaves the square-root branch sign open.
    for sign in (1.0, -1.0):
        def residual(p):
            return np.asarray([_predict_params(r, p, sign) for r in train_rows]) - y
        result = least_squares(residual, x0=np.array([3.0, 0.5, 0.2]), bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]), max_nfev=10000)
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x.copy(), sign)
    _params = (best[1], best[2])
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params[0], _params[1]))