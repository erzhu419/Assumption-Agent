import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(log_params):
        a_fast, a_slow, tau_fast, gap = np.exp(log_params)
        tau_slow = tau_fast + gap
        gain = -a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)
        return u * gain - y

    result = least_squares(residual, np.log([2.0, 3.0, 0.5, 12.0]), ftol=1e-11, xtol=1e-11, gtol=1e-11)
    a_fast, a_slow, tau_fast, gap = np.exp(result.x)
    _params = tuple(map(float, (a_fast, a_slow, tau_fast, tau_fast + gap)))


def predict(row):
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = float(row['H'])
    gain = -a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)
    return float(row['u_now']) * gain
