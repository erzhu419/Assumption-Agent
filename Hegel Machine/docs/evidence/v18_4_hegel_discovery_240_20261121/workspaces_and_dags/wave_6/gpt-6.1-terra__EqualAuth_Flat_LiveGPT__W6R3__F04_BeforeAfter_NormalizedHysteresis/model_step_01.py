import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    un = np.array([r['u_now'] / 0.02 for r in train_rows], dtype=float)
    up = np.array([r['u_past'] / 0.02 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        gain, log_tau, k_eq = p
        return gain * (un - up) * np.exp(-h / np.exp(log_tau)) + k_eq * un - y

    candidates = [least_squares(residual, [1.0, math.log(t), 0.5],
                                bounds=([-np.inf, -12.0, -np.inf],
                                        [np.inf, 12.0, np.inf]),
                                max_nfev=3000)
                  for t in (0.1, 0.5, 2.0)]
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x
    return _params

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 0.02
    up = row['u_past'] / 0.02
    return float(gain * (un - up) * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * un)
