import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(max(0.0, u))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    base, sink, branch = np.asarray([_features(r) for r in rows], dtype=float).T
    y = np.asarray([r['y'] for r in rows], dtype=float)

    def residual(p):
        vmax, rho, coeff = p
        return vmax * (1.0 - rho * sink) * base + coeff * branch - y

    _params = least_squares(residual, [8.0, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row) -> float:
    base, sink, branch = _features(row)
    vmax, rho, coeff = _params
    return float(vmax * (1.0 - rho * sink) * base + coeff * branch)
