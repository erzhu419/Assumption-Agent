import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    af, ass, tf, ts = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (af * (1.0 - np.exp(-h / tf)) - ass * (1.0 - np.exp(-h / ts)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    starts = [np.array([2.0, 1.0, 0.5, 5.0]), np.array([1.0, 2.0, 1.0, 10.0]), np.array([3.0, 2.0, 0.2, 20.0])]
    sols = [least_squares(residual, x0, bounds=([0.0, 0.0, 1e-4, 1e-4], [100.0, 100.0, 1e3, 1e3]), max_nfev=20000) for x0 in starts]
    best = min(sols, key=lambda s: float(np.dot(s.fun, s.fun)))
    _params = best.x
    return _params.copy()

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _params))