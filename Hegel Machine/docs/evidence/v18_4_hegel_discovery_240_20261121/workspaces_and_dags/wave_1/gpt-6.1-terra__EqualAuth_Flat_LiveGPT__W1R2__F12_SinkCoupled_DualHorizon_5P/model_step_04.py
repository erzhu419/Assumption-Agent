import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    return u, s, h

def _model(p, u, s, h):
    vmax, rho, c_sqrt, amp_fast, amp_slow, log_tf, log_ts = p
    tf, ts = np.exp(log_tf), np.exp(log_ts)
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * np.sqrt(np.maximum(u, 0.0))
    adaptation = u * (amp_fast * (-np.expm1(-h / tf)) - amp_slow * (-np.expm1(-h / ts)))
    return base + adaptation

def fit(train_rows):
    global _params
    u, s, h = _values(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [-20.0, -3.0, -20.0, -20.0, -20.0, np.log(0.02), np.log(0.5)]
    upper = [20.0, 3.0, 20.0, 20.0, 20.0, np.log(15.0), np.log(200.0)]
    best = None
    for tf in (0.2, 1.0, 3.0):
        for ts in (5.0, 15.0, 40.0):
            p0 = [2.0, 0.7, 0.2, 1.0, 2.0, np.log(tf), np.log(ts)]
            result = least_squares(lambda p: _model(p, u, s, h) - y, p0, bounds=(lower, upper), max_nfev=4000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    u, s, h = _values([row])
    return float(_model(_params, u, s, h)[0])