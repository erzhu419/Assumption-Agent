import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_values(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (-np.expm1(-H / tau_fast)) - amp_slow * (-np.expm1(-H / tau_slow)))

    def residuals(p):
        return predict_values(p) - y

    starts = [
        [1.2, 0.5, 0.8, 10.0],
        [1.5, 1.0, 1.0, 20.0],
        [2.0, 0.2, 1.0, 5.0],
        [1.0, 2.0, 0.8, 30.0],
    ]
    fits = [least_squares(residuals, x0, bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 1e3, 20.0, 1e4]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: np.sum(result.fun ** 2))
    _params = np.asarray(best.x, dtype=float)
    return {'amp_fast': float(_params[0]), 'tau_fast': float(_params[1]), 'amp_slow': float(_params[2]), 'tau_slow': float(_params[3])}

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, tau_fast, amp_slow, tau_slow = _params
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (-math.expm1(-H / tau_fast)) - amp_slow * (-math.expm1(-H / tau_slow))))