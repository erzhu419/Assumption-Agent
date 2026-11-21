import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    d = u - np.array([r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * d * np.exp(-h / p[1]) + p[2] * u - y
    candidates = [least_squares(residual, [-2.0, tau, 0.27], bounds=([-np.inf, 1e-5, -np.inf], [np.inf, np.inf, np.inf]), max_nfev=5000) for tau in (0.2, 0.7, 2.0)]
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    gain, tau, k_eq = _params
    return float(gain * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / tau) + k_eq * row['u_now'])