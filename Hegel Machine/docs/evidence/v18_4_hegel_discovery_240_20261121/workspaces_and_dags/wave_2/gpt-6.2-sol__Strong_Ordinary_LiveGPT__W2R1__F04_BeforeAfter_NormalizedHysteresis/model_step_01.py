import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau_scale = 1.0

def fit(train_rows):
    global _params
    data = np.array([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, observed = data.T
    now = now / _tau_scale
    delta = past / _tau_scale - now
    def residual(params):
        gain, rate, equilibrium = params
        return equilibrium * now + gain * delta * np.exp(-rate * horizon) - observed
    _params = least_squares(residual, [1.0, 1.0, 0.25], bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, equilibrium = _params
    now = row['u_now'] / _tau_scale
    past = row['u_past'] / _tau_scale
    return float(equilibrium * now + gain * (past - now) * math.exp(-rate * row['H']))
