import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, rho, c_sqrt, loss, threshold = p
    return vmax * (u / (1.2 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u) - loss * u * max(0.0, s - threshold)

def fit(train_rows):
    global _params
    samples = [(float(r['u_now']), float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residuals(p):
        return np.array([_law(u, s, p) - y for u, s, y in samples])
    result = least_squares(residuals, [7.0, 0.7, 0.3, 1.5, 2.5], bounds=([0.0, 0.0, -np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))