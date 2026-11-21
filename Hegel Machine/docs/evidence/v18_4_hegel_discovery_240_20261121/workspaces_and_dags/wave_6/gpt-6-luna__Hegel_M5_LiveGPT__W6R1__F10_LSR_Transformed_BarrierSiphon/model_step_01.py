import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 3.0
_PARAMS = None

def _features(row):
    un = float(row['u_now']) / _TAU
    sg = float(row['s_grad'])
    return un, sg

def _eval(row, p):
    j0, beta_bar, rho, c_sqrt = p
    un, sg = _features(row)
    inhibition = 1.0 - rho * sg / (1.0 + sg)
    return j0 * un * math.exp(-beta_bar / (0.5 + un)) * inhibition + c_sqrt * math.sqrt(max(un, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_eval(r, p) for r in rows]) - y
    starts = ([1.0, 1.0, 0.70, 0.1], [3.0, 2.0, 0.70, 0.3], [0.5, 0.2, 0.4, 0.5])
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.05, 0.0], [100.0, 100.0, 0.95, 100.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_eval(row, _PARAMS))