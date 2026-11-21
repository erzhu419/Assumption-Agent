import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.maximum(0.0, np.array([r['u_now'] for r in train_rows], dtype=float))
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    branch = np.sqrt(u)
    def residual(p):
        vmax, rho, gain = p
        return vmax * (1.0 - rho * sink) * base + gain * branch - y
    _params = least_squares(residual, [8.0, 0.7, 0.3], bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf])).x
    return _params

def predict(row) -> float:
    vmax, rho, gain = _params
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + gain * math.sqrt(u))