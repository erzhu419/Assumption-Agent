import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s))
            * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(
        residual,
        x0=np.array([3.0, 0.3, 0.5, 0.5, 10.0]),
        bounds=(np.array([0.0, 0.0, 1e-4, 0.0, 1e-3]),
                np.array([100.0, 1.0, 100.0, 100.0, 1000.0])),
        max_nfev=20000
    )
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))