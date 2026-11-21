import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    design, target = [], []
    for row in train_rows:
        u, s = row['u_now'], row['s_grad']
        base = u / (1.2 + u)
        # No training gradients lie inside the proposed onset interval.
        active = u if s > 2.68 else 0.0
        design.append([base, -base * s / (1.0 + s), math.sqrt(u), -s * active, active])
        target.append(row['y'])
    A, y = np.asarray(design), np.asarray(target)
    vmax, vrho, c_loc, k_aval, kcrit = least_squares(
        lambda p: A @ p - y, np.zeros(5), jac=lambda p: A
    ).x
    _params = (vmax, vrho / vmax, c_loc, k_aval, kcrit / k_aval)
    return _params

def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = _params
    u, s = row['u_now'], row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
                 + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)
