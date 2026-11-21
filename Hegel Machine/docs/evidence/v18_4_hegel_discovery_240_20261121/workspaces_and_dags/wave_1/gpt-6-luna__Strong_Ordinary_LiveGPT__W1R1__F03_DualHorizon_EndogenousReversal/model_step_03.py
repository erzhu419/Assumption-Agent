import numpy as np
from scipy.optimize import least_squares
import math

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    h = np.array([float(r['H']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    # Optimize log-parameters to keep both amplitudes and time constants positive.
    def residual(q):
        af, ass, tf, ts = np.exp(q)
        pred = u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
        return pred - y

    starts = [
        np.log([1.5, 1.5, 0.5, 12.0]),
        np.log([2.0, 1.0, 1.0, 20.0]),
        np.log([1.0, 1.0, 0.2, 8.0]),
    ]
    fits = [least_squares(residual, s, max_nfev=20000) for s in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _params = np.exp(best.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    af, ass, tf, ts = _params
    u = float(row['u_now'])
    h = float(row['H'])
    return float(u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))))