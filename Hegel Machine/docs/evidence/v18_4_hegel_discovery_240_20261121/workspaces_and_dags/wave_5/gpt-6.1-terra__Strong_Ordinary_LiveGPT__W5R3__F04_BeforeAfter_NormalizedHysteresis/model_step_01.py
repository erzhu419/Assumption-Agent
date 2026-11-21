import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    un = np.array([r['u_now'] / 0.02 for r in train_rows], dtype=float)
    dp = np.array([(r['u_now'] - r['u_past']) / 0.02 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * dp * np.exp(-h / np.exp(p[1])) + p[2] * un - y
    fits = [least_squares(residual, [g, np.log(t), 0.3], bounds=([-np.inf, -8, -np.inf], [np.inf, 8, np.inf]), max_nfev=3000) for t in (0.2, 0.6, 2.0) for g in (-3.0, 3.0)]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 0.02
    dp = (row['u_now'] - row['u_past']) / 0.02
    return float(gain * dp * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
