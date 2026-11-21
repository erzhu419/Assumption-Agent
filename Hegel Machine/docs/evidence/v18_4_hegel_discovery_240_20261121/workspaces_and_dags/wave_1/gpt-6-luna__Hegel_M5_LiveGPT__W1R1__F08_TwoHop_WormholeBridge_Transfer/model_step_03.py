import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # tau_scale is 1.0 for this task
    return un, s

def _predict_params(p, row):
    vmax, rho, c = p
    u, s = _features(row)
    base = u / (1.2 + u)
    return vmax * (1.0 - rho * s / (1.0 + s)) * base + c * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    # Fit only on labeled training observations.
    rows = [r for r in rows if 'y' in r]
    if not rows:
        raise ValueError('fit requires labeled rows containing y')
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residuals(p):
        return np.asarray([_predict_params(p, r) for r in rows]) - y
    result = least_squares(
        residuals,
        x0=np.array([6.5, 0.70, 0.1]),
        bounds=(np.array([1.0, 0.05, -2.0]), np.array([15.0, 0.95, 2.0])),
        max_nfev=20000,
    )
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('call fit(train_rows) before predict(row)')
    return float(_predict_params(_PARAMS, row))