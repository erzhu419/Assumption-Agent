import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        gain, log_tau, k_eq = p
        tau = math.exp(log_tau)
        return np.array([gain * ((r['u_now'] - r['u_past']) / 0.02) * math.exp(-r['H'] / tau) + k_eq * r['u_now'] / 0.02 - r['y'] for r in train_rows])
    best = None
    for tau0 in (0.1, 0.3, 0.7, 1.5, 4.0):
        result = least_squares(residual, [-2.0, math.log(tau0), 0.6], bounds=([-1000, -8, -1000], [1000, 8, 1000]))
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    gain, log_tau, k_eq = _params
    return float(gain * ((row['u_now'] - row['u_past']) / 0.02) * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * row['u_now'] / 0.02)
