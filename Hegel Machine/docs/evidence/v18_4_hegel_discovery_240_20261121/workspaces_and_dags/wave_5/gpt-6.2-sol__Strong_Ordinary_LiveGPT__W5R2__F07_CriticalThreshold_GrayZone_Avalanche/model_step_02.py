import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + c_loc * math.sqrt(u) - k_aval * u * max(0.0, s - s_crit)

def fit(train_rows):
    global _params
    data = [(float(r['u_now']), float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residuals(p):
        return np.array([_law(u, s, p) - y for u, s, y in data])
    _params = least_squares(residuals, [6.0, 0.7, 0.2, 1.0, 2.6], bounds=([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, 1.0, np.inf, np.inf, 2.78])).x

def predict(row) -> float:
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))