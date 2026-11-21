import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = u / (1.2 + u)
    return base, -base * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.asarray([_features(r) for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    # The fitted coefficients are vmax, vmax*rho, and the local branch amplitude.
    _params = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x

def predict(row) -> float:
    return float(np.dot(_features(row), _params))
