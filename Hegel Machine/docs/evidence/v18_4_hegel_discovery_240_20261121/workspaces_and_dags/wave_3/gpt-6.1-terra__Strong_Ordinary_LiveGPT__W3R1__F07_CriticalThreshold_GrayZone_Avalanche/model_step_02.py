import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, -20.0, 0.0, 2.44], [100.0, 2.0, 20.0, 100.0, 2.78])
    results = [least_squares(lambda p: _values(p, u, s) - y, [v, rho, c, k, t], bounds=bounds, max_nfev=10000, xtol=1e-13, ftol=1e-13, gtol=1e-13) for v, rho, c, k, t in [(8.0, 0.7, 1.0, 2.0, 2.55), (12.0, 0.7, 0.0, 2.0, 2.65)]]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad'])))