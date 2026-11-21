import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    amp_fast, amp_slow, tau_fast, tau_gap = p
    tau_slow = tau_fast + tau_gap
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    ys = np.asarray([float(r['y']) for r in rows])
    starts = [
        [1.3, 1.0, 0.3, 3.0],
        [1.5, 1.2, 0.5, 10.0],
        [1.2, 1.0, 1.0, 20.0],
        [2.0, 1.5, 0.2, 5.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(
            lambda p: np.asarray([_predict_params(r, p) for r in rows]) - ys,
            x0=np.asarray(x0, dtype=float),
            bounds=([0.0, 0.0, 1e-5, 1e-5], [20.0, 20.0, 100.0, 200.0]),
            max_nfev=20000,
        )
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _params = best.x.copy()
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params))