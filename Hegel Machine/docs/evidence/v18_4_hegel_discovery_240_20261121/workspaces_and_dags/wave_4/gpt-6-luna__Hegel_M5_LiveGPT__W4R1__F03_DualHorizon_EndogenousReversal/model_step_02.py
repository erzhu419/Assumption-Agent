import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    ys = np.asarray([float(r['y']) for r in rows])
    # Positive amplitudes and timescales; parameterize tau_slow > tau_fast.
    def unpack(z):
        return np.array([math.exp(z[0]), math.exp(z[1]), math.exp(z[2]), math.exp(z[1]) + math.exp(z[3])])
    def residual(z):
        p = unpack(z)
        return np.asarray([_model(r, p) for r in rows]) - ys
    best = None
    for start in ([math.log(1.0), math.log(1.0), math.log(0.5), math.log(4.0)],
                  [math.log(2.0), math.log(0.3), math.log(0.8), math.log(8.0)],
                  [math.log(1.5), math.log(2.0), math.log(1.0), math.log(10.0)]):
        result = least_squares(residual, np.asarray(start), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, unpack(result.x))
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _params))