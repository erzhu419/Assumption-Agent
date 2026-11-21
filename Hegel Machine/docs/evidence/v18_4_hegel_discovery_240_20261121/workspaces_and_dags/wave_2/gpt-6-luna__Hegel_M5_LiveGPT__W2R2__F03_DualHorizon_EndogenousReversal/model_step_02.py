import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    H = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    # Log-parameters keep amplitudes and timescales positive; tau_slow is
    # parameterized as tau_fast plus a positive separation.
    def unpack(z):
        af, ass, tf, gap = np.exp(z)
        return af, ass, tf, tf + gap
    def residual(z):
        p = unpack(z)
        return np.array([_value(r, p) for r in rows]) - ys
    starts = [
        np.log([1.5, 1.0, 0.5, 5.0]),
        np.log([2.0, 1.0, 1.0, 10.0]),
        np.log([1.0, 0.8, 0.2, 2.0]),
    ]
    results = [least_squares(residual, z, max_nfev=20000) for z in starts]
    best = min(results, key=lambda res: float(np.sum(res.fun ** 2)))
    af, ass, tf, ts = unpack(best.x)
    return {'amp_fast': float(af), 'amp_slow': float(ass), 'tau_fast': float(tf), 'tau_slow': float(ts)}

def predict(row):
    p = (float(_MODEL['amp_fast']), float(_MODEL['amp_slow']), float(_MODEL['tau_fast']), float(_MODEL['tau_slow']))
    return float(_value(row, p))

_MODEL = None
_original_fit = fit
def fit(train_rows):
    global _MODEL
    _MODEL = _original_fit(train_rows)
    return _MODEL
