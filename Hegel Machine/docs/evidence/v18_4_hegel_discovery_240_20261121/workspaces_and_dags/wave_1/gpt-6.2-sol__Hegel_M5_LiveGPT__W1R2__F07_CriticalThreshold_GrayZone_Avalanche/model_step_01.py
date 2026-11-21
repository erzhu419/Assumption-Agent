import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    branch = np.sqrt(u)

    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * sink) * base + c * branch - y

    result = least_squares(residual, [6.5, 0.70, 0.14], bounds=([3.836397, 0.658333, -0.822167], [9.163604, 0.748333, 1.108834]))
    if np.sqrt(np.mean(result.fun**2)) > 0.045:
        result = least_squares(residual, result.x, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x

def predict(row) -> float:
    vmax, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))