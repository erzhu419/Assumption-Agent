import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        gain, log_tau, k_eq = p
        tau = math.exp(log_tau)
        return np.array([gain * ((r['u_now'] - r['u_past']) / 5.0) * math.exp(-r['H'] / tau) + k_eq * r['u_now'] / 5.0 - r['y'] for r in train_rows])
    result = least_squares(residual, [1.5, math.log(5.0), 0.6], bounds=([-100.0, -10.0, -100.0], [100.0, 10.0, 100.0]))
    _params = result.x
    return _params

def predict(row):
    gain, log_tau, k_eq = _params
    return float(gain * ((row['u_now'] - row['u_past']) / 5.0) * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * row['u_now'] / 5.0)
