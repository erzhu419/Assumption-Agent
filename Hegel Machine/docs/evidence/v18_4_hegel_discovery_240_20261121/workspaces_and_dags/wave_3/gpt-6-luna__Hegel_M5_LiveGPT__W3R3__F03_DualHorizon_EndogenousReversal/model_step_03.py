import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _f(p, u, h):
    af, ass, tf, ts = p
    return u * (af * (-math.expm1(-h / tf)) - ass * (-math.expm1(-h / ts)))

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    # Log-parameterization guarantees positive amplitudes and timescales, with
    # tau_slow > tau_fast. Optimize from several plausible scale initializations.
    def decode(q):
        af, ass = np.exp(q[:2])
        tf = np.exp(q[2])
        ts = tf + np.exp(q[3])
        return af, ass, tf, ts

    def residual(q):
        af, ass, tf, ts = decode(q)
        pred = u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
        return pred - y

    best = None
    for tf0 in (0.2, 0.5, 1.0, 2.0):
        for gap0 in (2.0, 6.0, 14.0):
            q0 = np.log([1.2, 0.8, tf0, gap0])
            res = least_squares(residual, q0, max_nfev=5000)
            score = float(np.dot(res.fun, res.fun))
            if best is None or score < best[0]:
                best = (score, decode(res.x))
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_f(_params, float(row['u_now']), float(row['H'])))
