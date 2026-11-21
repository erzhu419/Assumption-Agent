import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau_scale = 1.0

def _model(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    u = float(row['u_now']) / _tau_scale
    h = float(row['H'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(logp):
        p = np.exp(logp)
        return np.asarray([_model(r, p) for r in rows]) - y
    best = None
    # Positive amplitudes and timescales; multi-start addresses nonlinear local minima.
    for start in ([1.0, 0.3, 1.0, 10.0], [2.0, 1.0, 1.0, 20.0], [1.0, 0.1, 2.0, 5.0], [3.0, 2.0, 2.0, 30.0]):
        result = least_squares(residual, np.log(start), bounds=(np.log([1e-8, 1e-5, 1e-8, 1e-4]), np.log([1e3, 1e3, 1e3, 1e4])), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, np.exp(result.x))
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_model(row, _params))