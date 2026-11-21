import numpy as np
from scipy.optimize import least_squares

_P = None

def _calc(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - np.exp(-h / tau_fast)) - amp_slow * (1.0 - np.exp(-h / tau_slow)))

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_calc(r, p) for r in rows]) - y
    starts = [[1.0, 0.2, 0.5, 10.0], [1.0, 1.0, 0.5, 20.0], [2.0, 0.3, 1.0, 30.0]]
    fits = [least_squares(residual, x0, bounds=([0, 1e-5, 0, 1e-5], [100, 1e4, 100, 1e4]), max_nfev=30000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.sum(z.fun ** 2)))
    _P = best.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_calc(row, _P))