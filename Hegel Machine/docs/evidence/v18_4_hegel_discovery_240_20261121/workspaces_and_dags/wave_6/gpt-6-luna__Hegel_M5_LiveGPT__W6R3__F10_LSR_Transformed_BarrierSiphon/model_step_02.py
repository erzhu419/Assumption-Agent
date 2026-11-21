import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, key):
    return float(row[key])

def _law(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = _value(row, 'u_now')
    s = _value(row, 's_grad')
    u_n = u  # tau_scale = 1.0
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([_value(r, 'y') for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_law(r, p) for r in rows]) - y
    starts = [np.array([5.0, 1.0, 0.70, 0.5]), np.array([8.0, 2.0, 0.70, 0.2]), np.array([3.0, 0.5, 0.70, 1.0])]
    lower = np.array([0.0, 0.0, 0.05, -20.0])
    upper = np.array([100.0, 30.0, 0.95, 20.0])
    results = [least_squares(residual, np.clip(x, lower + 1e-9, upper - 1e-9), bounds=(lower, upper), max_nfev=20000) for x in starts]
    _P = min(results, key=lambda z: float(np.dot(z.fun, z.fun))).x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_law(row, _P))
