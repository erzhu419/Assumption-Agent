import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([4.0, 0.3, 0.5, 0.5, 12.0]),
                           bounds=([0.0, 0.0, 0.0, 1e-4, 0.5001],
                                   [100.0, 1.0, 100.0, 5.0, 500.0]),
                           max_nfev=50000)
    _PARAMS = result.x.copy()
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _PARAMS))