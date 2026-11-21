import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-h / tau_slow))
    return base - adapt

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    p0 = np.array([3.0, 0.5, 0.5, 0.5, 10.0])
    result = least_squares(residual, p0, bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [100.0, 0.999, 100.0, 100.0, 1000.0]), max_nfev=50000)
    _params = result.x
    return {'vmax': float(_params[0]), 'rho': float(_params[1]), 'tau_fast': float(_params[2]), 'amp_adapt': float(_params[3]), 'tau_slow': float(_params[4])}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))