import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    lower = np.array([0.0, 0.0, 0.005, 0.0, 0.1])
    upper = np.array([20.0, 1.0, 10.0, 5.0, 200.0])
    starts = [
        [4.0, 0.4, 0.4, 0.5, 15.0],
        [3.0, 0.7, 1.0, 0.3, 25.0],
        [6.0, 0.2, 0.2, 0.8, 10.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in rows]) - y,
                            x0=np.asarray(x0, dtype=float), bounds=(lower, upper),
                            max_nfev=30000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _params = best.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))