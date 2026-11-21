import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    best = None
    # Positive amplitudes/timescales; multiple starts reduce local-minimum risk.
    for start in ([1.0, 1.0, 0.3, 10.0], [2.0, 1.0, 1.0, 20.0], [1.0, 2.0, 0.5, 5.0], [3.0, 2.0, 2.0, 30.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 1e-5, 1e-5], [100.0, 100.0, 1e4, 1e4]), max_nfev=20000)
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))