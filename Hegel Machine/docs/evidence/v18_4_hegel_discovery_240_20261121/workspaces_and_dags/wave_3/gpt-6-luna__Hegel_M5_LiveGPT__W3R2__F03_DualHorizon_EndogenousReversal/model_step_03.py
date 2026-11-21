import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    H = np.array([float(r['H']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def model(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (1.0 - np.exp(-H / tau_fast)) - amp_slow * (1.0 - np.exp(-H / tau_slow)))

    result = least_squares(
        lambda p: model(p) - y,
        x0=np.array([1.5, 0.3, 1.0, 12.0]),
        bounds=([0.0, 1e-4, 0.0, 1e-3], [20.0, 10.0, 20.0, 500.0]),
        max_nfev=50000
    )
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, tau_fast, amp_slow, tau_slow = _params
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow))))