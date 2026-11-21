import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    af, tf, ass, ts, rho = p
    u = float(row['u_now'])
    h = float(row['H'])
    s = float(row['s_grad'])
    temporal = af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts))
    sink = 1.0 - rho * s / (1.0 + s)
    return u * temporal * sink

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in rows]) - y
    best = None
    for start in ([2.0, 1.0, 2.0, 12.0, 0.2], [3.0, 2.0, 3.0, 20.0, 0.5], [1.5, 0.4, 2.5, 10.0, -0.2], [4.0, 3.0, 4.0, 30.0, 0.8]):
        res = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.02, 0.0, 0.05, -3.0], [20.0, 100.0, 20.0, 200.0, 3.0]), max_nfev=30000)
        score = float(np.mean(res.fun ** 2))
        if best is None or score < best[0]:
            best = (score, res.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(_params, row))