import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    H = np.asarray([float(r['H']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    scale = np.maximum(np.abs(u), 1e-12)
    z = y / scale

    # Parameterize positive time constants with tau_slow > tau_fast.
    def curve(p, h):
        af, ass, log_tf, log_gap = p
        tf = math.exp(log_tf)
        ts = tf + math.exp(log_gap)
        return af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts))

    best = None
    for tf0 in (0.05, 0.2, 0.5, 1.0, 2.0):
        for ts0 in (3.0, 8.0, 20.0, 50.0):
            if ts0 <= tf0:
                continue
            p0 = np.array([1.2, 1.0, math.log(tf0), math.log(ts0 - tf0)])
            result = least_squares(lambda p: curve(p, H) - z, p0,
                                   bounds=([-20.0, -20.0, math.log(1e-4), math.log(1e-4)],
                                           [20.0, 20.0, math.log(1e3), math.log(1e3)]),
                                   max_nfev=20000)
            loss = float(np.dot(result.fun, result.fun))
            if best is None or loss < best[0]:
                best = (loss, result.x)
    _params = best[1]
    af, ass, log_tf, log_gap = _params
    return {'amp_fast': float(af), 'amp_slow': float(ass),
            'tau_fast': math.exp(float(log_tf)),
            'tau_slow': math.exp(float(log_tf)) + math.exp(float(log_gap))}

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    af, ass, log_tf, log_gap = _params
    tf = math.exp(float(log_tf))
    ts = tf + math.exp(float(log_gap))
    h = float(row['H'])
    gain = af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts))
    return float(row['u_now']) * gain
