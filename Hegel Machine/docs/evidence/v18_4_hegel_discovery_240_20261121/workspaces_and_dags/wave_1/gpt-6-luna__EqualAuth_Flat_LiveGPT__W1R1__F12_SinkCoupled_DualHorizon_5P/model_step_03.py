import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        af, ass, tf, ts = p
        pred = u * (af * (1.0 - np.exp(-h / tf)) - ass * (1.0 - np.exp(-h / ts)))
        return pred - y
    best = None
    for p0 in ([1.0, 1.0, 0.4, 10.0], [2.0, 2.0, 1.0, 20.0], [1.0, 0.5, 0.2, 5.0], [3.0, 3.0, 2.0, 30.0]):
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 1e-4, 1e-3], [20.0, 20.0, 100.0, 500.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = tuple(float(v) for v in best.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    af, ass, tf, ts = _params
    h = float(row['H'])
    return float(row['u_now']) * (af * (1.0 - math.exp(-h / tf)) - ass * (1.0 - math.exp(-h / ts)))