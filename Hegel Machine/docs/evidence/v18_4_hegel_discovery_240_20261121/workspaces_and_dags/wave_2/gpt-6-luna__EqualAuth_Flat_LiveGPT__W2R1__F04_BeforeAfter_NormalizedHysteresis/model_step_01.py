import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    delta = float(row['u_now']) - float(row['u_past'])
    h = float(row['H'])
    x = float(row['x_now'])
    return delta, h, x

def fit(train_rows):
    rows = list(train_rows)
    if len(rows) < 3:
        raise ValueError('At least three rows are required to fit gain, relaxation time, and state coefficient.')
    def residual(p):
        gain, log_tau, state_coef = p
        tau = math.exp(log_tau)
        return np.asarray([gain * (d) * math.exp(-h / tau) + state_coef * x - float(r['y'])
                           for r in rows for d, h, x in [_features(r)]], dtype=float)
    initial = np.array([-1.5, math.log(0.8), 1.0])
    result = least_squares(residual, initial, max_nfev=5000)
    return tuple(float(v) for v in result.x)

def predict(row):
    gain, log_tau, state_coef = fit(_FIT_ROWS)
    delta, h, x = _features(row)
    return float(gain * delta * math.exp(-h / math.exp(log_tau)) + state_coef * x)
