import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _unpack(z):
    a, b, fast, gap = np.exp(z)
    return a, b, fast, fast + gap

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(z):
        a, b, fast, slow = _unpack(z)
        return u * (-a * np.expm1(-h / fast) + b * np.expm1(-h / slow)) - y

    best = None
    for fast, gap in [(0.4, 6.0), (0.2, 2.0), (0.8, 15.0)]:
        result = least_squares(residual, np.log([1.7, 2.3, fast, gap]), bounds=(-12.0, 12.0), ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=600)
        if best is None or result.cost < best.cost:
            best = result
    _params = _unpack(best.x)

def predict(row) -> float:
    a, b, fast, slow = _params
    h = float(row['H'])
    return float(row['u_now'] * (-a * math.expm1(-h / fast) + b * math.expm1(-h / slow)))
