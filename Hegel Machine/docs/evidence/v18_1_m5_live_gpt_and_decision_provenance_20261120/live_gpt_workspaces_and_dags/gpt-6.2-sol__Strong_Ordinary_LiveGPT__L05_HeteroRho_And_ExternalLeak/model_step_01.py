import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])  # normalized by the supplied tau scale of 1.0
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), -u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([_features(r) for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    # Coefficients are vmax, vmax*rho, and leak_rate.
    _params = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x
    return _params

def predict(row) -> float:
    return math.fsum(p * f for p, f in zip(_params, _features(row)))
