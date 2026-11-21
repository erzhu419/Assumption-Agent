import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.02
_params = None

def _features(row):
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    # Barrier-activated flux, inhibited by the sink gradient, plus a square-root baseline.
    return u, s

def _eval(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    scale = max(float(np.std(y)), 1.0)
    def residual(p):
        return np.asarray([_eval(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5, 0.1]),
                           bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]),
                           x_scale='jac', max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_eval(_params, row))