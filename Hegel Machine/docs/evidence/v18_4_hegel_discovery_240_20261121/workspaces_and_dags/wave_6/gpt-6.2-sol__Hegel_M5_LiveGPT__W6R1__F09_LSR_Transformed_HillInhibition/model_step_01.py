import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _law(u, s, p) - y
    best = None
    for start in ((5., 1., 1., .1), (4., .2, .3, .05), (10., 4., 2., .3)):
        result = least_squares(residual, start, bounds=([0., 1e-9, 0., 0.], np.inf), x_scale='jac', max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if math.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
        if best is not None and best.cost < 1e-12:
            break
    _params = best.x

def predict(row):
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))
