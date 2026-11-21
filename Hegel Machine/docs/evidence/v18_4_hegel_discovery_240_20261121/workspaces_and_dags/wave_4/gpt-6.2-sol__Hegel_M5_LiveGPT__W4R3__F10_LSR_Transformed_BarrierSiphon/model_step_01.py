import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.02
_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] / TAU for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - y

    _params = least_squares(residual, [4.0, 1.0, 0.70, 0.2],
                            bounds=([0.0, 0.0, 0.05, -np.inf],
                                    [np.inf, np.inf, 0.95, np.inf]),
                            ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    j0, beta, rho, c = _params
    u = row['u_now'] / TAU
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))
