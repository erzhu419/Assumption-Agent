import math
import numpy as np
from scipy.optimize import least_squares

_tau = 1.0
_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float).T
    u = u / _tau
    q = u * u

    def residual(p):
        vmax, k_half, alpha_inh, c_leak = p
        return vmax * q / ((k_half + q) * (1.0 + alpha_inh * s)) - c_leak * u - y

    starts = ([6.0, 0.6, 0.8, 0.3], [10.0, 2.0, 2.0, 0.6])
    result = min((least_squares(residual, p, bounds=([0.0, 1e-10, 0.0, 0.0], np.inf), x_scale='jac') for p in starts), key=lambda r: r.cost)
    _params = result.x

def predict(row) -> float:
    vmax, k_half, alpha_inh, c_leak = _params
    u = float(row['u_now']) / _tau
    q = math.pow(u, 2)
    return float(vmax * q / ((k_half + q) * (1.0 + alpha_inh * float(row['s_grad']))) - c_leak * u)
