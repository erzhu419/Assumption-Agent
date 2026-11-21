import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    # Positive amplitudes and ordered time constants: tau_slow > tau_fast.
    def predict_params(z):
        af, ass = np.exp(z[0]), np.exp(z[1])
        tf, ts = np.exp(z[2]), np.exp(z[2]) + np.exp(z[3])
        return u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    starts = [np.log([1.0, 1.0, tf, ts_delta]) for tf, ts_delta in [(0.5, 5.0), (1.0, 10.0), (2.0, 20.0)]]
    best = None
    for z0 in starts:
        res = least_squares(lambda z: predict_params(z) - y, z0, max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    z = best.x
    return {'amp_fast': float(np.exp(z[0])), 'amp_slow': float(np.exp(z[1])), 'tau_fast': float(np.exp(z[2])), 'tau_slow': float(np.exp(z[2]) + np.exp(z[3]))}

def predict(row):
    p = fit_params
    h = float(row['H'])
    return float(row['u_now']) * (p['amp_fast'] * (1.0 - math.exp(-h / p['tau_fast'])) - p['amp_slow'] * (1.0 - math.exp(-h / p['tau_slow'])))

fit_params = None
# The evaluation harness fits once before prediction; cache fitted parameters through fit().
_fit_impl = fit
def fit(train_rows):
    global fit_params
    fit_params = _fit_impl(train_rows)
    return fit_params
