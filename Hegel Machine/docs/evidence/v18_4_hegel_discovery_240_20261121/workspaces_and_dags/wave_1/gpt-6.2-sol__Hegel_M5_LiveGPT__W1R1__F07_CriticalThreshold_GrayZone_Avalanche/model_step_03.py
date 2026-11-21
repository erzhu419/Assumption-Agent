import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p, sqrt, positive):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * sqrt(u) - k_aval * positive(s - s_crit) * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        return _law(u, s, p, np.sqrt, lambda z: np.maximum(0.0, z)) - y
    result = least_squares(residual, [6.5, 0.70, 0.15, 1.0, 2.56], bounds=([1.0, 0.05, -2.0, 0.0, 0.0], [15.0, 0.95, 2.0, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    return float(_law(row['u_now'], row['s_grad'], _params, math.sqrt, lambda z: max(0.0, z)))
