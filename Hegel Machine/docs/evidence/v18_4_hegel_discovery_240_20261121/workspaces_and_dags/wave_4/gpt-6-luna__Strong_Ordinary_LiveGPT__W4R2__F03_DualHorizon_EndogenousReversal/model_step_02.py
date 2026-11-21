import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    # Positive amplitudes and time constants are optimized in log space.
    def predict_values(q):
        amp_fast, tau_fast, amp_slow, tau_slow = np.exp(q)
        return u * (amp_fast * (-np.expm1(-H / tau_fast))
                    - amp_slow * (-np.expm1(-H / tau_slow)))

    starts = [np.log([1.5, 0.5, 1.0, 12.0]),
              np.log([2.0, 1.0, 1.0, 20.0]),
              np.log([1.0, 0.2, 0.5, 8.0])]
    best = None
    for q0 in starts:
        result = least_squares(lambda q: predict_values(q) - y, q0,
                               bounds=(np.log([1e-5, 1e-4, 1e-5, 1e-3]),
                                       np.log([100.0, 100.0, 100.0, 1e4])),
                               max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = np.exp(best.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    amp_fast, tau_fast, amp_slow, tau_slow = _params
    H = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (-math.expm1(-H / tau_fast))
                      - amp_slow * (-math.expm1(-H / tau_slow))))