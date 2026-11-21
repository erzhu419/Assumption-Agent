import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, rho, branch, gain, critical = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + branch * math.sqrt(max(0.0, u)) - gain * u * max(0.0, s - critical)

def fit(train_rows):
    global _params
    rows = [(float(r['u_now']), float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residuals(p):
        return np.array([_law(u, s, p) - y for u, s, y in rows])
    _params = least_squares(residuals, [6.0, 0.7, 0.3, 1.5, 2.5], x_scale='jac').x

def predict(row) -> float:
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))