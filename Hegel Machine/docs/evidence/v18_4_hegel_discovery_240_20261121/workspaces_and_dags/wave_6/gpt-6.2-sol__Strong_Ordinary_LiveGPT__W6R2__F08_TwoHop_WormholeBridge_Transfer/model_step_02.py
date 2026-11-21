import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 1.0
    s = float(row['s_grad'])
    f = u / (1.2 + u)
    return f, -f * s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(row) for row in rows], dtype=float)
    y = np.array([row['y'] for row in rows], dtype=float)
    # Fit vmax, vmax*rho, and the signed square-root coefficient.
    _params = least_squares(lambda p: X @ p - y, np.zeros(3), jac=lambda p: X).x

def predict(row) -> float:
    return float(np.asarray(_features(row)) @ _params)
