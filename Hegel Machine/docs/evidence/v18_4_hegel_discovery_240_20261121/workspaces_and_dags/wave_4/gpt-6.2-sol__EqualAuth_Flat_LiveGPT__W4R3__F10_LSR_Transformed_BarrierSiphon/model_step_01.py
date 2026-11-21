import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, sink, root):
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] / 0.02 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    sink = s / (1.0 + s)
    root = np.sqrt(u)
    y = np.array([r['y'] for r in rows], dtype=float)
    _params = least_squares(lambda p: _law(p, u, sink, root) - y,
                            [3.0, 1.0, 0.5, 0.3],
                            bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf])).x

def predict(row) -> float:
    u = row['u_now'] / 0.02
    s = row['s_grad']
    j0, beta, rho, c = _params
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))