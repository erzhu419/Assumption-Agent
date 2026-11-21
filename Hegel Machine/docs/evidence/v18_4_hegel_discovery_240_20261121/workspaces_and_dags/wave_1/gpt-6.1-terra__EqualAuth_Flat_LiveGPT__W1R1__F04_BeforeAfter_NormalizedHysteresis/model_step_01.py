import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        gain, log_tau, k_eq = p
        tau = math.exp(log_tau)
        return np.array([gain * (r['u_now'] - r['u_past']) * math.exp(-r['H'] / tau) + k_eq * r['u_now'] - r['y'] for r in rows])
    fits = [least_squares(residual, [g, math.log(t), 0.5], bounds=([-100.0, -10.0, -100.0], [100.0, 10.0, 100.0])) for g in (1.0, 3.0, 6.0) for t in (0.2, 1.0, 5.0)]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x
    return _params

def predict(row):
    gain, log_tau, k_eq = _params
    return float(gain * (row['u_now'] - row['u_past']) * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * row['u_now'])