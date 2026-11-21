import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _evaluate(p, u, s, h) - y, [5.0, 0.7, 0.5, 1.5, 10.0], bounds=([0.0, 0.0, 0.001, 0.0, 0.01], [100.0, 2.0, 100.0, 100.0, 1000.0]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_evaluate(_params, float(row['u_now']), float(row['s_grad']), float(row['H'])))