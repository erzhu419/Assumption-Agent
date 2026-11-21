import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _unpack(z):
    a_fast, a_slow, tau_fast, gap = np.exp(z)
    return a_fast, a_slow, tau_fast, tau_fast + gap

def fit(train_rows):
    global _params
    rows = list(train_rows)
    h = np.array([r['H'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(z):
        af, a_s, tf, ts = _unpack(z)
        gain = -af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)
        return u * gain - y

    result = least_squares(residual, np.log([1.6, 3.0, 0.4, 14.0]), ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = _unpack(result.x)

def predict(row) -> float:
    af, a_s, tf, ts = _params
    h = float(row['H'])
    gain = -af * math.expm1(-h / tf) + a_s * math.expm1(-h / ts)
    return float(row['u_now'] * gain)
