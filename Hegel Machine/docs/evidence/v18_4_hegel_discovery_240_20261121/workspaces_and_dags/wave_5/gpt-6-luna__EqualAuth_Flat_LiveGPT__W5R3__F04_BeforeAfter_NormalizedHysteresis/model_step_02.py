import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    scale = 0.02
    un = float(row['u_now']) / scale
    up = float(row['u_past']) / scale
    return un, un - up, float(row['H'])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    un = np.array([_features(r)[0] for r in rows])
    delta = np.array([_features(r)[1] for r in rows])
    h = np.array([_features(r)[2] for r in rows])
    def residual(p):
        gain, log_tau, k_eq = p
        tau = math.exp(log_tau)
        return gain * delta * np.exp(-h / tau) + k_eq * un - y
    p0 = np.array([-2.0, math.log(0.5), 0.3])
    result = least_squares(residual, p0, max_nfev=20000)
    return {'gain': float(result.x[0]), 'tau_relax': float(math.exp(result.x[1])), 'k_eq': float(result.x[2])}

def predict(row):
    p = fit(_TRAIN_ROWS)
    un, delta, h = _features(row)
    return float(p['gain'] * delta * math.exp(-h / p['tau_relax']) + p['k_eq'] * un)
