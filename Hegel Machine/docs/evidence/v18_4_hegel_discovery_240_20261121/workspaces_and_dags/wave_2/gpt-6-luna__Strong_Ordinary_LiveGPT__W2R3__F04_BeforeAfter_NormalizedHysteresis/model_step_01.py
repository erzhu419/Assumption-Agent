import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    scale = 0.004
    un = float(row['u_now']) / scale
    up = float(row['u_past']) / scale
    H = float(row['H'])
    return un, up, H

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        gain, log_tau, k_eq = p
        tau = math.exp(log_tau)
        return np.array([gain * (un - up) * math.exp(-H / tau) + k_eq * un
                         for un, up, H in map(_features, rows)]) - y
    best = None
    for tau0 in (0.01, 0.1, 1.0, 10.0):
        result = least_squares(residual, [1.0, math.log(tau0), 0.0], max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    gain, log_tau, k_eq = best.x
    return {'gain': float(gain), 'tau_relax': float(math.exp(log_tau)), 'k_eq': float(k_eq)}

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else row
    un, up, H = _features(row)
    return float(p['gain'] * (un - up) * math.exp(-H / p['tau_relax']) + p['k_eq'] * un)
