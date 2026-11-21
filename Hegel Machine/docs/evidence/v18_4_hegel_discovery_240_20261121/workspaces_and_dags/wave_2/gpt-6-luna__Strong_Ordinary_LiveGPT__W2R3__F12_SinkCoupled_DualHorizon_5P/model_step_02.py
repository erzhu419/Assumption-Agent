import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    lower = np.array([0.0, 0.0, 0.005, 0.0, 0.05])
    upper = np.array([30.0, 0.999, 10.0, 10.0, 200.0])
    starts = [
        [4.0, 0.4, 0.5, 0.5, 15.0],
        [3.0, 0.7, 0.2, 0.4, 30.0],
        [6.0, 0.2, 1.0, 0.8, 10.0],
        [2.0, 0.5, 0.1, 0.2, 50.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
                            np.asarray(x0, dtype=float), bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _params = best[1]
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))