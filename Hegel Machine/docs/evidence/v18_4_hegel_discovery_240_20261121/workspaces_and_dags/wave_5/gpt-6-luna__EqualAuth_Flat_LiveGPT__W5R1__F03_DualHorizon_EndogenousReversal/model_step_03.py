import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def predict_params(p):
        amp_fast, amp_slow, tau_fast, tau_gap = p
        tau_slow = tau_fast + tau_gap
        return u * (amp_fast * (-np.expm1(-H / tau_fast)) - amp_slow * (-np.expm1(-H / tau_slow)))

    starts = [
        [1.0, 1.0, 1.0, 5.0],
        [1.5, 1.5, 0.5, 10.0],
        [2.0, 2.0, 2.0, 15.0],
        [1.0, 1.2, 0.2, 20.0],
    ]
    best = None
    for p0 in starts:
        result = least_squares(
            lambda p: predict_params(p) - y,
            p0,
            bounds=([0.0, 0.0, 1e-6, 1e-6], [20.0, 20.0, 100.0, 500.0]),
            max_nfev=20000,
        )
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, amp_slow, tau_fast, tau_gap = _PARAMS
    tau_slow = tau_fast + tau_gap
    u = float(row['u_now'])
    H = float(row['H'])
    return float(u * (amp_fast * (-math.expm1(-H / tau_fast)) - amp_slow * (-math.expm1(-H / tau_slow))))