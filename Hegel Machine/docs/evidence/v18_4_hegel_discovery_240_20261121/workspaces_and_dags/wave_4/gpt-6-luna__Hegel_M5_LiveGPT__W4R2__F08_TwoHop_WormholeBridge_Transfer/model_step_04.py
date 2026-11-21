import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    A, k, g0, g1 = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = A * (1.0 - math.exp(-k * u))
    # The fin gradient modulates the two-hop bridge response around the
    # baseline gradient; include a gradient-by-input interaction.
    return base + (s - 1.0) * (g0 + g1 * u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([3.5, 0.8, 0.0, 0.0]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _PARAMS))