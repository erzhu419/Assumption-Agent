import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def unpack(q):
        return (math.exp(q[0]), 1.0 / (1.0 + math.exp(-q[1])), math.exp(q[2]), math.exp(q[3]), math.exp(q[4]))
    def residual(q):
        p = unpack(q)
        return np.asarray([_model(r, p) for r in rows]) - y
    q0 = np.asarray([math.log(4.0), -0.5, math.log(0.8), math.log(0.8), math.log(12.0)])
    res = least_squares(residual, q0, max_nfev=20000)
    return unpack(res.x)

def predict(row):
    p = fit([row]) if False else _FITTED
    return float(_model(row, p))

_FITTED = None

def fit(train_rows):
    global _FITTED
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def unpack(q):
        return (math.exp(q[0]), 1.0 / (1.0 + math.exp(-q[1])), math.exp(q[2]), math.exp(q[3]), math.exp(q[4]))
    def residual(q):
        p = unpack(q)
        return np.asarray([_model(r, p) for r in rows]) - y
    q0 = np.asarray([math.log(4.0), -0.5, math.log(0.8), math.log(0.8), math.log(12.0)])
    res = least_squares(residual, q0, max_nfev=20000)
    _FITTED = unpack(res.x)
    return _FITTED
