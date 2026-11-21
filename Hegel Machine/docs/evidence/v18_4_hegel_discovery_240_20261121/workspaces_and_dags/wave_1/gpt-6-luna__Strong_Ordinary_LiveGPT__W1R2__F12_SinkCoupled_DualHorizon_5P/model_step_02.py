import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        af, tf, ass, ts = p
        pred = u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
        return pred - y

    # Positive rates and ordered time scales identify the fast/slow components.
    best = None
    for start in ([1.0, 1.0, 1.5, 12.0], [2.0, 2.0, 2.5, 20.0], [0.8, 0.5, 1.0, 8.0], [3.0, 4.0, 3.5, 30.0]):
        res = least_squares(residual, start,
                            bounds=([0.0, 1e-4, 0.0, 1e-3], [20.0, 100.0, 20.0, 300.0]),
                            max_nfev=20000)
        if res.x[3] > res.x[1] and (best is None or np.sum(res.fun**2) < best[0]):
            best = (np.sum(res.fun**2), res.x)
    if best is None:
        raise RuntimeError('Could not fit ordered dual-horizon adaptation parameters')
    _params = tuple(float(v) for v in best[1])
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    af, tf, ass, ts = _params
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts)))